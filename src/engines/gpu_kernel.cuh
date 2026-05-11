#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace quakecore {
namespace gpu_kernel {

struct __align__(16) NodePacked {
  float minx;
  float miny;
  float minz;
  int32_t c0;
  float maxx;
  float maxy;
  float maxz;
  int32_t c1;
};

struct __align__(16) LeafPacked {
  float minx;
  float miny;
  float minz;
  int32_t pad0;
  float maxx;
  float maxy;
  float maxz;
  int32_t pad1;
};

struct __align__(16) PlaneGpu {
  float nx;
  float ny;
  float nz;
  float dist;
};

struct DeviceStats {
  unsigned long long visited_nodes;
  unsigned long long visited_leafs;
  unsigned long long culled_nodes;
  unsigned long long accepted_leafs;
};

constexpr int kStackMax = 128;
constexpr int kMaxWarpsPerBlock = 32;
constexpr int kKernelMaxThreads = 256;
constexpr int kKernelMinBlocksPerSM = 4;

__device__ __forceinline__ bool PlanePass(const PlaneGpu& p, float minx, float miny, float minz,
                                          float maxx, float maxy, float maxz) {
  // Reject corner: choose box corner most likely to be on the negative side of the plane.
  // Matches CPU baseline (frustum.cpp): reject = mins where normal>=0 else maxs;
  // box is culled iff dot(n, reject) - dist > 0.
  const float rx = (p.nx >= 0.0f) ? minx : maxx;
  const float ry = (p.ny >= 0.0f) ? miny : maxy;
  const float rz = (p.nz >= 0.0f) ? minz : maxz;
  return (p.nx * rx + p.ny * ry + p.nz * rz - p.dist) <= 0.0f;
}

__global__ __launch_bounds__(kKernelMaxThreads, kKernelMinBlocksPerSM)
void TraversalKernel(int num_cameras, int num_nodes, int num_leafs, int root,
                                const NodePacked* __restrict__ nodes,
                                const LeafPacked* __restrict__ leafs,
                                const PlaneGpu* __restrict__ frustums,
                                DeviceStats* __restrict__ out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  unsigned long long lvn = 0;
  unsigned long long lvl = 0;
  unsigned long long lcn = 0;
  unsigned long long lal = 0;

  for (int cam = tid; cam < num_cameras; cam += stride) {
    PlaneGpu pl[6];
    const float4* fsrc = reinterpret_cast<const float4*>(frustums + cam * 6);
#pragma unroll
    for (int i = 0; i < 6; ++i) {
      const float4 v = __ldg(fsrc + i);
      pl[i].nx = v.x;
      pl[i].ny = v.y;
      pl[i].nz = v.z;
      pl[i].dist = v.w;
    }

    int stack[kStackMax];
    int sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
      const int idx = stack[--sp];
      if (idx < 0 || idx >= num_nodes) {
        continue;
      }
      lvn++;

      const float4 a = __ldg(reinterpret_cast<const float4*>(nodes + idx));
      const float4 b = __ldg(reinterpret_cast<const float4*>(nodes + idx) + 1);
      const float minx = a.x;
      const float miny = a.y;
      const float minz = a.z;
      const int c0 = __float_as_int(a.w);
      const float maxx = b.x;
      const float maxy = b.y;
      const float maxz = b.z;
      const int c1 = __float_as_int(b.w);

      bool inside = true;
#pragma unroll
      for (int i = 0; i < 6; ++i) {
        if (!PlanePass(pl[i], minx, miny, minz, maxx, maxy, maxz)) {
          inside = false;
          break;
        }
      }

      if (!inside) {
        lcn++;
        continue;
      }

      // Process both children. Order doesn't affect counts.
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int child = (c == 0) ? c0 : c1;
        if (child >= 0) {
          // Bounds-check happens on pop. Stack guarded for safety; typical Quake BSP depth << kStackMax.
          if (sp < kStackMax) {
            stack[sp++] = child;
          }
        } else {
          const int leaf_index = -1 - child;
          if (leaf_index >= 0 && leaf_index < num_leafs) {
            lvl++;
            const float4 la = __ldg(reinterpret_cast<const float4*>(leafs + leaf_index));
            const float4 lb = __ldg(reinterpret_cast<const float4*>(leafs + leaf_index) + 1);
            const float lminx = la.x;
            const float lminy = la.y;
            const float lminz = la.z;
            const float lmaxx = lb.x;
            const float lmaxy = lb.y;
            const float lmaxz = lb.z;
            bool linside = true;
#pragma unroll
            for (int i = 0; i < 6; ++i) {
              if (!PlanePass(pl[i], lminx, lminy, lminz, lmaxx, lmaxy, lmaxz)) {
                linside = false;
                break;
              }
            }
            if (linside) {
              lal++;
            }
          }
        }
      }
    }
  }

  // Block-wide reduction: warp shuffle -> shared mem -> warp shuffle -> atomicAdd
  const unsigned int mask = 0xffffffffu;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    lvn += __shfl_xor_sync(mask, lvn, off);
    lvl += __shfl_xor_sync(mask, lvl, off);
    lcn += __shfl_xor_sync(mask, lcn, off);
    lal += __shfl_xor_sync(mask, lal, off);
  }

  __shared__ unsigned long long s_vn[kMaxWarpsPerBlock];
  __shared__ unsigned long long s_vl[kMaxWarpsPerBlock];
  __shared__ unsigned long long s_cn[kMaxWarpsPerBlock];
  __shared__ unsigned long long s_al[kMaxWarpsPerBlock];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int n_warps = (blockDim.x + 31) >> 5;

  if (lane == 0) {
    s_vn[warp_id] = lvn;
    s_vl[warp_id] = lvl;
    s_cn[warp_id] = lcn;
    s_al[warp_id] = lal;
  }
  __syncthreads();

  if (warp_id == 0) {
    unsigned long long vn = (lane < n_warps) ? s_vn[lane] : 0ull;
    unsigned long long vl = (lane < n_warps) ? s_vl[lane] : 0ull;
    unsigned long long cn = (lane < n_warps) ? s_cn[lane] : 0ull;
    unsigned long long al = (lane < n_warps) ? s_al[lane] : 0ull;

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      vn += __shfl_xor_sync(mask, vn, off);
      vl += __shfl_xor_sync(mask, vl, off);
      cn += __shfl_xor_sync(mask, cn, off);
      al += __shfl_xor_sync(mask, al, off);
    }

    if (lane == 0) {
      atomicAdd(&out->visited_nodes, vn);
      atomicAdd(&out->visited_leafs, vl);
      atomicAdd(&out->culled_nodes, cn);
      atomicAdd(&out->accepted_leafs, al);
    }
  }
}

}  // namespace gpu_kernel
}  // namespace quakecore

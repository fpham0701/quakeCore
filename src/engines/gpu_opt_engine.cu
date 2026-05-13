#include "quakecore/engines.hpp"
#include "quakecore/frustum.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace quakecore {
namespace {

inline void CheckCuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

// AABB node packed to two 16-byte chunks
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

// AABB leaf packed to two 16-byte chunks
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

// final stats for culling step
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

// microkernel to determine if a `PlaneGpu` passes culling
// box is culled iff dot(n, reject) - dist > 0
__device__ __forceinline__ bool PlanePass(const PlaneGpu &p, float minx,
                                          float miny, float minz, float maxx,
                                          float maxy, float maxz) {
  const float rx = (p.nx >= 0.0f) ? minx : maxx;
  const float ry = (p.ny >= 0.0f) ? miny : maxy;
  const float rz = (p.nz >= 0.0f) ? minz : maxz;
  return (p.nx * rx + p.ny * ry + p.nz * rz - p.dist) <= 0.0f;
}

// AABB tree traversal kernel
__global__ __launch_bounds__(
    kKernelMaxThreads,
    kKernelMinBlocksPerSM) void TraversalKernel(int num_cameras, int num_nodes,
                                                int num_leafs, int root,
                                                const NodePacked
                                                    *__restrict__ nodes,
                                                const LeafPacked
                                                    *__restrict__ leafs,
                                                const PlaneGpu
                                                    *__restrict__ frustums,
                                                DeviceStats *__restrict__ out) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  unsigned long long lvn = 0;
  unsigned long long lvl = 0;
  unsigned long long lcn = 0;
  unsigned long long lal = 0;

  // camera per thread
  for (int cam = tid; cam < num_cameras; cam += stride) {
    PlaneGpu pl[6];
    const float4 *fsrc = reinterpret_cast<const float4 *>(frustums + cam * 6);

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

      const float4 a = __ldg(reinterpret_cast<const float4 *>(nodes + idx));
      const float4 b = __ldg(reinterpret_cast<const float4 *>(nodes + idx) + 1);
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

// process both children, ordering won't affect final counts
      #pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int child = (c == 0) ? c0 : c1;
        if (child >= 0) {
          // bounds check happens on pop, this is only for safety not
          // correctness
          if (sp < kStackMax) {
            stack[sp++] = child;
          }
        } else {
          const int leaf_index = -1 - child;
          if (leaf_index >= 0 && leaf_index < num_leafs) {
            lvl++;
            const float4 la =
                __ldg(reinterpret_cast<const float4 *>(leafs + leaf_index));
            const float4 lb =
                __ldg(reinterpret_cast<const float4 *>(leafs + leaf_index) + 1);
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

  // reduction path: warp shuffle -> shared mem -> warp shuffle -> atomicAdd
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

} // namespace

// host caller for traversal kernel
TraversalStats RunGpuOptimizedTraversal(const BspData &bsp,
                                        const std::vector<Camera> &cameras,
                                        const int block_size,
                                        const Frustum* override_frustum) {
  if (bsp.models.empty()) {
    throw std::runtime_error("BSP has no model data");
  }
  if (cameras.empty()) {
    return TraversalStats{};
  }

  const int num_cameras = static_cast<int>(cameras.size());
  const int num_nodes = static_cast<int>(bsp.nodes.size());
  const int num_leafs = static_cast<int>(bsp.leafs.size());
  const int root = bsp.models.front().headnode[0];

  // reduce allocations by using a single host buffer and single device buffer
  // layout: [nodes][leafs][frustums][stats] with 256-byte alignment between
  // regions
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) / a * a; };
  const size_t bytes_nodes =
      sizeof(NodePacked) * static_cast<size_t>(num_nodes);
  const size_t bytes_leafs =
      sizeof(LeafPacked) * static_cast<size_t>(num_leafs);
  const size_t bytes_frus =
      sizeof(PlaneGpu) * static_cast<size_t>(num_cameras) * 6;
  const size_t bytes_stats = sizeof(DeviceStats);
  const size_t off_nodes = 0;
  const size_t off_leafs = align_up(off_nodes + bytes_nodes, 256);
  const size_t off_frus = align_up(off_leafs + bytes_leafs, 256);
  const size_t off_stats = align_up(off_frus + bytes_frus, 256);
  const size_t total_bytes = off_stats + bytes_stats;

  // cudaMallocHost not used here because we do not get the added throughput
  // benefit we are paying for at typical BSP sizes (<= ~50MB transferred)
  std::vector<unsigned char> h_buf_storage(total_bytes);
  unsigned char *h_buf = h_buf_storage.data();

  auto *h_nodes = reinterpret_cast<NodePacked *>(h_buf + off_nodes);
  auto *h_leafs = reinterpret_cast<LeafPacked *>(h_buf + off_leafs);
  auto *h_frustums = reinterpret_cast<PlaneGpu *>(h_buf + off_frus);

  for (int i = 0; i < num_nodes; ++i) {
    const auto &n = bsp.nodes[static_cast<size_t>(i)];
    h_nodes[i].minx = n.mins[0];
    h_nodes[i].miny = n.mins[1];
    h_nodes[i].minz = n.mins[2];
    h_nodes[i].c0   = n.children[0];
    h_nodes[i].maxx = n.maxs[0];
    h_nodes[i].maxy = n.maxs[1];
    h_nodes[i].maxz = n.maxs[2];
    h_nodes[i].c1   = n.children[1];
  }

  for (int i = 0; i < num_leafs; ++i) {
    const auto &l = bsp.leafs[static_cast<size_t>(i)];
    h_leafs[i].minx = l.mins[0];
    h_leafs[i].miny = l.mins[1];
    h_leafs[i].minz = l.mins[2];
    h_leafs[i].pad0 = 0;
    h_leafs[i].maxx = l.maxs[0];
    h_leafs[i].maxy = l.maxs[1];
    h_leafs[i].maxz = l.maxs[2];
    h_leafs[i].pad1 = 0;
  }

  for (int i = 0; i < num_cameras; ++i) {
    const Frustum f = override_frustum ? *override_frustum : BuildFrustum(cameras[static_cast<size_t>(i)]);
    for (int p = 0; p < 6; ++p) {
      const auto &pl = f.planes[static_cast<size_t>(p)];
      h_frustums[static_cast<size_t>(i) * 6 + p].nx = pl.normal.x;
      h_frustums[static_cast<size_t>(i) * 6 + p].ny = pl.normal.y;
      h_frustums[static_cast<size_t>(i) * 6 + p].nz = pl.normal.z;
      h_frustums[static_cast<size_t>(i) * 6 + p].dist = pl.dist;
    }
  }

  // legacy stream used here b/c standard stream creation + cudaMallocAsync pool
  // init costs are not worth it for a single-shot kernel launch
  // TODO: change to standard stream if integrating into Quake engine
  unsigned char *d_buf = nullptr;
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&d_buf), total_bytes),
            "cudaMalloc buf failed");

  auto *d_nodes = reinterpret_cast<NodePacked *>(d_buf + off_nodes);
  auto *d_leafs = reinterpret_cast<LeafPacked *>(d_buf + off_leafs);
  auto *d_frustums = reinterpret_cast<PlaneGpu *>(d_buf + off_frus);
  auto *d_stats = reinterpret_cast<DeviceStats *>(d_buf + off_stats);

  // single memcpy between device and host buffers
  CheckCuda(cudaMemcpy(d_buf, h_buf, off_stats, cudaMemcpyHostToDevice),
            "cudaMemcpy staging failed");
  CheckCuda(cudaMemset(d_stats, 0, bytes_stats), "cudaMemset stats failed");

  int threads = (block_size > 0) ? block_size : 256;
  if (threads < 32)
    threads = 32;
  if (threads > kKernelMaxThreads)
    threads = kKernelMaxThreads;
  threads = ((threads + 31) / 32) * 32;

  // explicitly avoiding oversubscription by matching threads executing to
  // actual workload; subscribing more than this would end up paying scheduling
  // cost while leaving most threads idle
  int sm_count = 1;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  const int blocks_for_work = (num_cameras + threads - 1) / threads;
  const int min_blocks_for_fill = sm_count;
  int blocks = blocks_for_work;
  if (blocks < min_blocks_for_fill && num_cameras >= min_blocks_for_fill * 32) {
    blocks = min_blocks_for_fill;
  }

  TraversalKernel<<<blocks, threads>>>(num_cameras, num_nodes, num_leafs, root,
                                       d_nodes, d_leafs, d_frustums, d_stats);
  CheckCuda(cudaPeekAtLastError(), "kernel launch failed");

  DeviceStats h{};
  CheckCuda(
      cudaMemcpy(&h, d_stats, sizeof(DeviceStats), cudaMemcpyDeviceToHost),
      "cudaMemcpy result failed");
  CheckCuda(cudaFree(d_buf), "cudaFree buf failed");

  TraversalStats out{};
  out.visited_nodes = h.visited_nodes;
  out.visited_leafs = h.visited_leafs;
  out.culled_nodes = h.culled_nodes;
  out.accepted_leafs = h.accepted_leafs;
  return out;
}

} // namespace quakecore

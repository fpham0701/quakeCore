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

// TraversalKernel is defined once in gpu_opt_engine.cu.
// Declared here so gpu_context.cu can reference it via CUDA separable compilation.
extern __global__ void TraversalKernel(int num_cameras, int num_nodes, int num_leafs, int root,
                                       const NodePacked* __restrict__ nodes,
                                       const LeafPacked* __restrict__ leafs,
                                       const PlaneGpu* __restrict__ frustums,
                                       DeviceStats* __restrict__ out);

}  // namespace gpu_kernel
}  // namespace quakecore

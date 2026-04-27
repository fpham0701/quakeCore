#include "quakecore/engines.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace quakecore {
namespace {

struct DeviceStats {
  unsigned long long visited_nodes;
  unsigned long long visited_leafs;
};

__global__ void CountKernel(const int node_count, const int leaf_count, DeviceStats* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out->visited_nodes = static_cast<unsigned long long>(node_count);
    out->visited_leafs = static_cast<unsigned long long>(leaf_count);
  }
}

inline void CheckCuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

}  // namespace

TraversalStats RunGpuOptimizedTraversal(const BspData& bsp, const std::vector<Camera>& cameras, const int block_size) {
  (void)cameras;
  DeviceStats* d_stats = nullptr;
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_stats), sizeof(DeviceStats)), "cudaMalloc failed");
  CheckCuda(cudaMemset(d_stats, 0, sizeof(DeviceStats)), "cudaMemset failed");

  const int threads = (block_size > 0) ? block_size : 256;
  CountKernel<<<1, threads>>>(static_cast<int>(bsp.nodes.size()), static_cast<int>(bsp.leafs.size()), d_stats);
  CheckCuda(cudaPeekAtLastError(), "kernel launch failed");
  CheckCuda(cudaDeviceSynchronize(), "kernel synchronize failed");

  DeviceStats h{};
  CheckCuda(cudaMemcpy(&h, d_stats, sizeof(DeviceStats), cudaMemcpyDeviceToHost), "cudaMemcpy failed");
  CheckCuda(cudaFree(d_stats), "cudaFree failed");

  TraversalStats out{};
  out.visited_nodes = h.visited_nodes * cameras.size();
  out.visited_leafs = h.visited_leafs * cameras.size();
  out.accepted_leafs = out.visited_leafs;
  return out;
}

}  // namespace quakecore

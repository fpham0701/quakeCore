#include "quakecore/engines.hpp"
#include "quakecore/frustum.hpp"
#include "engines/gpu_kernel.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace quakecore {
namespace {

using gpu_kernel::NodePacked;
using gpu_kernel::LeafPacked;
using gpu_kernel::PlaneGpu;
using gpu_kernel::DeviceStats;
using gpu_kernel::kKernelMaxThreads;

inline void CheckCuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

}  // namespace

TraversalStats RunGpuOptimizedTraversal(const BspData& bsp, const std::vector<Camera>& cameras,
                                        const int block_size) {
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

  // One contiguous host staging buffer and one device buffer.
  // Layout: [nodes][leafs][frustums][stats] with 256-byte alignment between regions.
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) / a * a; };
  const size_t bytes_nodes = sizeof(NodePacked) * static_cast<size_t>(num_nodes);
  const size_t bytes_leafs = sizeof(LeafPacked) * static_cast<size_t>(num_leafs);
  const size_t bytes_frus = sizeof(PlaneGpu) * static_cast<size_t>(num_cameras) * 6;
  const size_t bytes_stats = sizeof(DeviceStats);
  const size_t off_nodes = 0;
  const size_t off_leafs = align_up(off_nodes + bytes_nodes, 256);
  const size_t off_frus = align_up(off_leafs + bytes_leafs, 256);
  const size_t off_stats = align_up(off_frus + bytes_frus, 256);
  const size_t total_bytes = off_stats + bytes_stats;

  // Plain pageable host buffer. cudaMallocHost (pinned) was tested but its one-time
  // registration cost (~30ms+) outweighs the H2D throughput gain at the workload sizes
  // this benchmark exercises (<= ~50MB total transfer).
  std::vector<unsigned char> h_buf_storage(total_bytes);
  unsigned char* h_buf = h_buf_storage.data();

  auto* h_nodes = reinterpret_cast<NodePacked*>(h_buf + off_nodes);
  auto* h_leafs = reinterpret_cast<LeafPacked*>(h_buf + off_leafs);
  auto* h_frustums = reinterpret_cast<PlaneGpu*>(h_buf + off_frus);

  for (int i = 0; i < num_nodes; ++i) {
    const auto& n = bsp.nodes[static_cast<size_t>(i)];
    h_nodes[i].minx = static_cast<float>(n.mins[0]);
    h_nodes[i].miny = static_cast<float>(n.mins[1]);
    h_nodes[i].minz = static_cast<float>(n.mins[2]);
    h_nodes[i].c0 = n.children[0];
    h_nodes[i].maxx = static_cast<float>(n.maxs[0]);
    h_nodes[i].maxy = static_cast<float>(n.maxs[1]);
    h_nodes[i].maxz = static_cast<float>(n.maxs[2]);
    h_nodes[i].c1 = n.children[1];
  }
  for (int i = 0; i < num_leafs; ++i) {
    const auto& l = bsp.leafs[static_cast<size_t>(i)];
    h_leafs[i].minx = static_cast<float>(l.mins[0]);
    h_leafs[i].miny = static_cast<float>(l.mins[1]);
    h_leafs[i].minz = static_cast<float>(l.mins[2]);
    h_leafs[i].pad0 = 0;
    h_leafs[i].maxx = static_cast<float>(l.maxs[0]);
    h_leafs[i].maxy = static_cast<float>(l.maxs[1]);
    h_leafs[i].maxz = static_cast<float>(l.maxs[2]);
    h_leafs[i].pad1 = 0;
  }
  for (int i = 0; i < num_cameras; ++i) {
    const Frustum f = BuildFrustum(cameras[static_cast<size_t>(i)]);
    for (int p = 0; p < 6; ++p) {
      const auto& pl = f.planes[static_cast<size_t>(p)];
      h_frustums[static_cast<size_t>(i) * 6 + p].nx = pl.normal.x;
      h_frustums[static_cast<size_t>(i) * 6 + p].ny = pl.normal.y;
      h_frustums[static_cast<size_t>(i) * 6 + p].nz = pl.normal.z;
      h_frustums[static_cast<size_t>(i) * 6 + p].dist = pl.dist;
    }
  }

  // Use the default (legacy) stream. Stream creation + the cudaMallocAsync pool init
  // each carry standalone cost on the first CUDA call of a process; the legacy stream
  // path avoids both. Memcpy/launch ordering on the default stream is sufficient for
  // a single-shot launch like this one.
  unsigned char* d_buf = nullptr;
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_buf), total_bytes), "cudaMalloc buf failed");

  auto* d_nodes = reinterpret_cast<NodePacked*>(d_buf + off_nodes);
  auto* d_leafs = reinterpret_cast<LeafPacked*>(d_buf + off_leafs);
  auto* d_frustums = reinterpret_cast<PlaneGpu*>(d_buf + off_frus);
  auto* d_stats = reinterpret_cast<DeviceStats*>(d_buf + off_stats);

  // One H2D copy of the whole staging region (nodes + leafs + frustums).
  CheckCuda(cudaMemcpy(d_buf, h_buf, off_stats, cudaMemcpyHostToDevice), "cudaMemcpy staging failed");
  CheckCuda(cudaMemset(d_stats, 0, bytes_stats), "cudaMemset stats failed");

  int threads = (block_size > 0) ? block_size : 256;
  if (threads < 32) threads = 32;
  if (threads > kKernelMaxThreads) threads = kKernelMaxThreads;
  threads = ((threads + 31) / 32) * 32;  // round up to warp

  // Don't over-subscribe: with one thread per camera and a grid-stride loop,
  // launching many more threads than cameras leaves most threads idle while still
  // paying scheduling cost. Pad to fill SMs only when the natural grid is smaller
  // than half the device — which is when occupancy actually limits performance.
  int sm_count = 1;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  const int blocks_for_work = (num_cameras + threads - 1) / threads;
  const int min_blocks_for_fill = sm_count;
  int blocks = blocks_for_work;
  if (blocks < min_blocks_for_fill && num_cameras >= min_blocks_for_fill * 32) {
    blocks = min_blocks_for_fill;
  }

  gpu_kernel::TraversalKernel<<<blocks, threads>>>(num_cameras, num_nodes, num_leafs, root,
                                                   d_nodes, d_leafs, d_frustums, d_stats);
  CheckCuda(cudaPeekAtLastError(), "kernel launch failed");

  DeviceStats h{};
  CheckCuda(cudaMemcpy(&h, d_stats, sizeof(DeviceStats), cudaMemcpyDeviceToHost),
            "cudaMemcpy result failed");
  CheckCuda(cudaFree(d_buf), "cudaFree buf failed");

  TraversalStats out{};
  out.visited_nodes = h.visited_nodes;
  out.visited_leafs = h.visited_leafs;
  out.culled_nodes = h.culled_nodes;
  out.accepted_leafs = h.accepted_leafs;
  return out;
}

}  // namespace quakecore

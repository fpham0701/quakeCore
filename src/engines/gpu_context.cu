#include "quakecore/gpu_context.hpp"
#include "quakecore/frustum.hpp"

#include "engines/gpu_kernel.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
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

inline std::size_t AlignUp(std::size_t v, std::size_t a) {
  return (v + a - 1) / a * a;
}

}  // namespace

struct GpuTraversalContext {
  // Map/launch-shape (immutable after Create).
  int num_nodes{0};
  int num_leafs{0};
  int root{0};
  int max_views_per_frame{0};
  int threads{0};
  int blocks{0};

  // One device allocation, sliced into regions.
  unsigned char* d_buf{nullptr};
  std::size_t off_nodes{0};
  std::size_t off_leafs{0};
  std::size_t off_frus{0};
  std::size_t off_stats{0};
  std::size_t total_bytes{0};
  std::size_t bytes_frus{0};
  std::size_t bytes_stats{0};

  // Pinned host buffers.
  PlaneGpu* h_frustums{nullptr};  // size = max_views_per_frame * 6
  DeviceStats* h_stats{nullptr};

  // Stream for graph capture / launch.
  cudaStream_t stream{nullptr};

  // Graph state (lazy-captured on first GpuContextRun call).
  cudaGraphExec_t graph_exec{nullptr};
  bool graph_captured{false};
};

GpuTraversalContext* GpuContextCreate(const BspData& bsp,
                                      const int max_views_per_frame,
                                      const int block_size) {
  if (bsp.models.empty()) {
    throw std::runtime_error("GpuContextCreate: BSP has no model data");
  }
  if (max_views_per_frame <= 0) {
    throw std::runtime_error("GpuContextCreate: max_views_per_frame must be > 0");
  }

  // Warm the CUDA primary context. Absorbs ~150 ms first-call init here so the
  // first Run does not pay it.
  CheckCuda(cudaFree(0), "cudaFree(0) warm-up failed");

  auto* ctx = new GpuTraversalContext{};
  ctx->num_nodes = static_cast<int>(bsp.nodes.size());
  ctx->num_leafs = static_cast<int>(bsp.leafs.size());
  ctx->root = bsp.models.front().headnode[0];
  ctx->max_views_per_frame = max_views_per_frame;

  // Region layout: [nodes][leafs][frustums][stats], each 256-byte aligned.
  const std::size_t bytes_nodes = sizeof(NodePacked) * static_cast<std::size_t>(ctx->num_nodes);
  const std::size_t bytes_leafs = sizeof(LeafPacked) * static_cast<std::size_t>(ctx->num_leafs);
  ctx->bytes_frus =
      sizeof(PlaneGpu) * static_cast<std::size_t>(max_views_per_frame) * 6;
  ctx->bytes_stats = sizeof(DeviceStats);
  ctx->off_nodes = 0;
  ctx->off_leafs = AlignUp(ctx->off_nodes + bytes_nodes, 256);
  ctx->off_frus = AlignUp(ctx->off_leafs + bytes_leafs, 256);
  ctx->off_stats = AlignUp(ctx->off_frus + ctx->bytes_frus, 256);
  ctx->total_bytes = ctx->off_stats + ctx->bytes_stats;

  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&ctx->d_buf), ctx->total_bytes),
            "cudaMalloc d_buf failed");

  // Pack and upload nodes/leafs in one H2D. Pageable host staging:
  // cudaMallocHost registration cost outweighs H2D throughput gain at the
  // BSP sizes this harness exercises, and the upload happens once per Create.
  std::vector<unsigned char> staging(ctx->off_frus);
  auto* h_nodes = reinterpret_cast<NodePacked*>(staging.data() + ctx->off_nodes);
  auto* h_leafs = reinterpret_cast<LeafPacked*>(staging.data() + ctx->off_leafs);

  for (int i = 0; i < ctx->num_nodes; ++i) {
    const auto& n = bsp.nodes[static_cast<std::size_t>(i)];
    h_nodes[i].minx = static_cast<float>(n.mins[0]);
    h_nodes[i].miny = static_cast<float>(n.mins[1]);
    h_nodes[i].minz = static_cast<float>(n.mins[2]);
    h_nodes[i].c0 = n.children[0];
    h_nodes[i].maxx = static_cast<float>(n.maxs[0]);
    h_nodes[i].maxy = static_cast<float>(n.maxs[1]);
    h_nodes[i].maxz = static_cast<float>(n.maxs[2]);
    h_nodes[i].c1 = n.children[1];
  }
  for (int i = 0; i < ctx->num_leafs; ++i) {
    const auto& l = bsp.leafs[static_cast<std::size_t>(i)];
    h_leafs[i].minx = static_cast<float>(l.mins[0]);
    h_leafs[i].miny = static_cast<float>(l.mins[1]);
    h_leafs[i].minz = static_cast<float>(l.mins[2]);
    h_leafs[i].pad0 = 0;
    h_leafs[i].maxx = static_cast<float>(l.maxs[0]);
    h_leafs[i].maxy = static_cast<float>(l.maxs[1]);
    h_leafs[i].maxz = static_cast<float>(l.maxs[2]);
    h_leafs[i].pad1 = 0;
  }
  CheckCuda(cudaMemcpy(ctx->d_buf, staging.data(), ctx->off_frus, cudaMemcpyHostToDevice),
            "cudaMemcpy nodes/leafs failed");

  // Pinned host frustum buffer (rewritten every Run, used by captured H2D node).
  CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&ctx->h_frustums), ctx->bytes_frus),
            "cudaMallocHost frustums failed");
  std::memset(ctx->h_frustums, 0, ctx->bytes_frus);

  // Pinned host stats result slot (written by captured D2H node).
  CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&ctx->h_stats), ctx->bytes_stats),
            "cudaMallocHost stats failed");
  std::memset(ctx->h_stats, 0, ctx->bytes_stats);

  int threads = (block_size > 0) ? block_size : 256;
  if (threads < 32) threads = 32;
  if (threads > kKernelMaxThreads) threads = kKernelMaxThreads;
  threads = ((threads + 31) / 32) * 32;
  ctx->threads = threads;

  int sm_count = 1;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  const int blocks_for_work = (max_views_per_frame + threads - 1) / threads;
  int blocks = blocks_for_work;
  if (blocks < sm_count && max_views_per_frame >= sm_count * 32) {
    blocks = sm_count;
  }
  if (blocks < 1) blocks = 1;
  ctx->blocks = blocks;

  // Non-default stream (graph capture cannot use the legacy default stream).
  CheckCuda(cudaStreamCreate(&ctx->stream), "cudaStreamCreate failed");

  return ctx;
}

TraversalStats GpuContextRun(GpuTraversalContext* ctx,
                             const std::vector<Camera>& views_for_frame) {
  if (!ctx) {
    throw std::runtime_error("GpuContextRun: null context");
  }
  if (static_cast<int>(views_for_frame.size()) != ctx->max_views_per_frame) {
    throw std::runtime_error(
        "GpuContextRun: views_for_frame.size() (" +
        std::to_string(views_for_frame.size()) +
        ") must equal max_views_per_frame (" +
        std::to_string(ctx->max_views_per_frame) + ")");
  }

  // Build all max_views_per_frame * 6 frustum planes into the pinned host buffer.
  // Order: views_for_frame[i].plane[j] -> h_frustums[i*6 + j].
  for (int i = 0; i < ctx->max_views_per_frame; ++i) {
    const Frustum f = BuildFrustum(views_for_frame[static_cast<std::size_t>(i)]);
    for (int p = 0; p < 6; ++p) {
      const auto& pl = f.planes[static_cast<std::size_t>(p)];
      PlaneGpu& dst = ctx->h_frustums[static_cast<std::size_t>(i) * 6 + p];
      dst.nx = pl.normal.x;
      dst.ny = pl.normal.y;
      dst.nz = pl.normal.z;
      dst.dist = pl.dist;
    }
  }

  auto* d_frustums = reinterpret_cast<PlaneGpu*>(ctx->d_buf + ctx->off_frus);
  auto* d_stats = reinterpret_cast<DeviceStats*>(ctx->d_buf + ctx->off_stats);
  auto* d_nodes = reinterpret_cast<NodePacked*>(ctx->d_buf + ctx->off_nodes);
  auto* d_leafs = reinterpret_cast<LeafPacked*>(ctx->d_buf + ctx->off_leafs);

  if (!ctx->graph_captured) {
    // Capture H2D + memset + kernel + D2H into a graph.
    CheckCuda(cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal),
              "cudaStreamBeginCapture failed");

    cudaError_t inner_err = cudaSuccess;
    auto record = [&](cudaError_t e) {
      if (inner_err == cudaSuccess) inner_err = e;
    };
    record(cudaMemcpyAsync(d_frustums, ctx->h_frustums, ctx->bytes_frus,
                           cudaMemcpyHostToDevice, ctx->stream));
    record(cudaMemsetAsync(d_stats, 0, ctx->bytes_stats, ctx->stream));
    gpu_kernel::TraversalKernel<<<ctx->blocks, ctx->threads, 0, ctx->stream>>>(
        ctx->max_views_per_frame, ctx->num_nodes, ctx->num_leafs, ctx->root,
        d_nodes, d_leafs, d_frustums, d_stats);
    record(cudaPeekAtLastError());
    record(cudaMemcpyAsync(ctx->h_stats, d_stats, ctx->bytes_stats,
                           cudaMemcpyDeviceToHost, ctx->stream));

    cudaGraph_t graph = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(ctx->stream, &graph);
    if (inner_err != cudaSuccess) {
      if (graph) cudaGraphDestroy(graph);
      throw std::runtime_error(std::string("graph capture body failed: ") +
                               cudaGetErrorString(inner_err));
    }
    if (end_err != cudaSuccess) {
      if (graph) cudaGraphDestroy(graph);
      throw std::runtime_error(std::string("cudaStreamEndCapture failed: ") +
                               cudaGetErrorString(end_err));
    }

    cudaError_t inst_err =
        cudaGraphInstantiate(&ctx->graph_exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (inst_err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaGraphInstantiate failed: ") +
                               cudaGetErrorString(inst_err));
    }
    ctx->graph_captured = true;
  }

  CheckCuda(cudaGraphLaunch(ctx->graph_exec, ctx->stream), "cudaGraphLaunch failed");
  CheckCuda(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize failed");

  TraversalStats out{};
  out.visited_nodes = ctx->h_stats->visited_nodes;
  out.visited_leafs = ctx->h_stats->visited_leafs;
  out.culled_nodes = ctx->h_stats->culled_nodes;
  out.accepted_leafs = ctx->h_stats->accepted_leafs;
  return out;
}

void GpuContextDestroy(GpuTraversalContext* ctx) {
  if (!ctx) return;
  if (ctx->graph_exec) {
    cudaGraphExecDestroy(ctx->graph_exec);
    ctx->graph_exec = nullptr;
  }
  if (ctx->stream) {
    cudaStreamDestroy(ctx->stream);
    ctx->stream = nullptr;
  }
  if (ctx->h_frustums) {
    cudaFreeHost(ctx->h_frustums);
    ctx->h_frustums = nullptr;
  }
  if (ctx->h_stats) {
    cudaFreeHost(ctx->h_stats);
    ctx->h_stats = nullptr;
  }
  if (ctx->d_buf) {
    cudaFree(ctx->d_buf);
    ctx->d_buf = nullptr;
  }
  delete ctx;
}

}  // namespace quakecore

#include "quakecore/gpu_context.hpp"

#include "engines/gpu_kernel.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace quakecore {

struct GpuTraversalContext {
  // Filled in by Task 3.
};

GpuTraversalContext* GpuContextCreate(const BspData& /*bsp*/,
                                      int /*max_views_per_frame*/,
                                      int /*block_size*/) {
  throw std::runtime_error("GpuContextCreate: not implemented yet");
}

TraversalStats GpuContextRun(GpuTraversalContext* /*ctx*/,
                             const std::vector<Camera>& /*views_for_frame*/) {
  throw std::runtime_error("GpuContextRun: not implemented yet");
}

void GpuContextDestroy(GpuTraversalContext* /*ctx*/) {
  // Stub.
}

}  // namespace quakecore

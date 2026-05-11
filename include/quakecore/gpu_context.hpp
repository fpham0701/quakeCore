#pragma once

#include <vector>

#include "quakecore/types.hpp"

namespace quakecore {

// Opaque — full definition lives in src/engines/gpu_context.cu.
struct GpuTraversalContext;

// One-time setup. Allocates and uploads packed nodes/leafs once.
// Captures a CUDA Graph sized for max_views_per_frame on the first
// GpuContextRun call (lazy, so the cost is visible to the caller).
GpuTraversalContext* GpuContextCreate(const BspData& bsp,
                                      int max_views_per_frame,
                                      int block_size);

// One frame of work. views_for_frame.size() must equal the
// max_views_per_frame passed to Create.
// Returns stats summed over just this frame's views.
TraversalStats GpuContextRun(GpuTraversalContext* ctx,
                             const std::vector<Camera>& views_for_frame);

// Releases device memory, CUDA Graph, frustum staging buffer.
// Does not destroy the CUDA primary context (process-wide, reused).
void GpuContextDestroy(GpuTraversalContext* ctx);

}  // namespace quakecore

#pragma once

#include <vector>

#include "quakecore/types.hpp"

namespace quakecore {

/**
 * Sequential pointer-style baseline traversal.
 */
TraversalStats RunBaselineTraversal(const BspData& bsp, const std::vector<Camera>& cameras);

/**
 * Flattened SoA + OpenMP/AVX traversal.
 */
TraversalStats RunCpuOptimizedTraversal(const BspData& bsp, const std::vector<Camera>& cameras, int threads);

/**
 * CUDA optimized traversal (or runtime fallback if CUDA disabled).
 */
TraversalStats RunGpuOptimizedTraversal(const BspData& bsp, const std::vector<Camera>& cameras, int block_size);

}  // namespace quakecore

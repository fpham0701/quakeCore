#pragma once

#include <vector>

#include "quakecore/types.hpp"

namespace quakecore {

/**
 * Sequential pointer-style baseline traversal.
 */
TraversalStats RunBaselineTraversal(const BspData& bsp, const std::vector<Camera>& cameras,
                                    const Frustum* override_frustum = nullptr);

/**
 * Flattened SoA + OpenMP/AVX traversal.
 */
TraversalStats RunCpuOptimizedTraversal(const BspData& bsp, const std::vector<Camera>& cameras, int threads,
                                        const Frustum* override_frustum = nullptr);

/**
 * CUDA optimized traversal.
 * Requires CUDA toolchain and runtime support at build/run time.
 */
TraversalStats RunGpuOptimizedTraversal(const BspData& bsp, const std::vector<Camera>& cameras, int block_size,
                                        const Frustum* override_frustum = nullptr);

}  // namespace quakecore

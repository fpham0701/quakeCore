#pragma once

#include <vector>

#include "quakecore/types.hpp"

namespace quakecore {

/**
 * Generate deterministic synthetic camera path for benchmark frames.
 */
std::vector<Camera> GenerateCameraPath(const BspData& bsp, int frames, uint64_t seed);

}  // namespace quakecore

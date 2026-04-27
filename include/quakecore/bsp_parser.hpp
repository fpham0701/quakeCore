#pragma once

#include "quakecore/types.hpp"

namespace quakecore {

/**
 * Parse a Quake 1 BSP (version 29) file.
 * Throws std::runtime_error on parse or validation failures.
 */
BspData ParseBspFile(const std::string& path);

}  // namespace quakecore

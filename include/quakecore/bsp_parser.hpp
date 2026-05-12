#pragma once

#include "quakecore/types.hpp"

namespace quakecore {

/**
 * Parse a Quake 1 BSP file. Supports two on-disk formats:
 *   - v29 (header version == 29): 16-bit indices, int16 quantized AABB bounds.
 *   - BSP2 (header version == ASCII "BSP2", little-endian 0x32505342):
 *     32-bit indices, float AABB bounds.
 *
 * Both formats populate the same widened in-memory BspData. v29 bounds
 * are widened from int16 to float at load time (exact: every int16 is
 * representable as float). Children are sign-extended from int16 to int32.
 *
 * Throws std::runtime_error on unsupported magic, truncated lumps, or
 * lump element-size mismatch. The BSP29a/2PSB intermediate format and
 * Half-Life BSP (version 30) are explicitly unsupported and rejected.
 */
BspData ParseBspFile(const std::string& path);

}  // namespace quakecore

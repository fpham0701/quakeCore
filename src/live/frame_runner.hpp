#pragma once
#include "quakecore/types.hpp"
#include "quakecore_live/frame_protocol.h"
#include <cstdint>

namespace quakecore::live {

struct FrameResult {
  uint64_t frame_id;
  uint64_t game_ns;
  uint64_t baseline_ns;
  uint64_t cpu_ns;
  uint64_t gpu_ns;
  uint32_t game_visited_nodes;
  uint32_t game_accepted_leafs;
  uint64_t qc_visited_nodes;
  uint64_t qc_accepted_leafs;
  bool     internal_parity_ok;
};

FrameResult RunOneFrame(const BspData& bsp, const QcfpFramePacket& pkt,
                        int threads, int gpu_block_size);

}  // namespace quakecore::live

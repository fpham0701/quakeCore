#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "quakecore/bsp_parser.hpp"
#include "quakecore/camera_path.hpp"
#include "quakecore/types.hpp"

namespace quakecore {

inline BenchmarkConfig ParseArgs(const int argc, char** argv) {
  BenchmarkConfig cfg{};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--map" && i + 1 < argc) {
      cfg.map_path = argv[++i];
    } else if (a == "--frames" && i + 1 < argc) {
      cfg.frames = std::stoi(argv[++i]);
    } else if (a == "--threads" && i + 1 < argc) {
      cfg.threads = std::stoi(argv[++i]);
    } else if (a == "--block-size" && i + 1 < argc) {
      cfg.gpu_block_size = std::stoi(argv[++i]);
    } else if (a == "--seed" && i + 1 < argc) {
      cfg.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
    } else if (a == "--help") {
      std::cout << "Usage: --map <file.bsp> [--frames N] [--threads N] [--block-size N] [--seed N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + a);
    }
  }
  if (cfg.map_path.empty()) {
    throw std::runtime_error("Missing required argument: --map <file.bsp>");
  }
  return cfg;
}

inline void PrintSummary(const char* name, const TraversalStats& stats, const double seconds, const int frames) {
  std::cout << "engine=" << name << "\n";
  std::cout << "frames=" << frames << "\n";
  std::cout << "elapsed_s=" << seconds << "\n";
  std::cout << "fps=" << (frames / seconds) << "\n";
  std::cout << "visited_nodes=" << stats.visited_nodes << "\n";
  std::cout << "visited_leafs=" << stats.visited_leafs << "\n";
  std::cout << "culled_nodes=" << stats.culled_nodes << "\n";
  std::cout << "accepted_leafs=" << stats.accepted_leafs << "\n";
}

}  // namespace quakecore

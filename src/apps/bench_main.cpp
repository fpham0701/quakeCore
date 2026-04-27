#include "app_common.hpp"
#include "quakecore/engines.hpp"

#include <fstream>
#include <iomanip>
#include <map>

namespace {

struct EngineResult {
  quakecore::TraversalStats stats{};
  double seconds{0.0};
};

template <typename Fn>
EngineResult RunTimed(Fn&& fn) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  const auto stats = fn();
  const auto t1 = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> dt = t1 - t0;
  return EngineResult{stats, dt.count()};
}

bool CheckParity(const quakecore::TraversalStats& base, const quakecore::TraversalStats& other) {
  return base.accepted_leafs == other.accepted_leafs && base.visited_nodes == other.visited_nodes;
}

void PrintLine(const std::string& name, const EngineResult& r, const int frames, const double baseline_seconds) {
  const double fps = static_cast<double>(frames) / r.seconds;
  const double speedup = baseline_seconds / r.seconds;
  std::cout << std::left << std::setw(12) << name << " "
            << "time_s=" << std::setw(10) << std::setprecision(6) << std::fixed << r.seconds << " "
            << "fps=" << std::setw(12) << std::setprecision(3) << fps << " "
            << "speedup=" << std::setw(10) << std::setprecision(3) << speedup << " "
            << "visited_nodes=" << r.stats.visited_nodes << " "
            << "accepted_leafs=" << r.stats.accepted_leafs << "\n";
}

void WriteCsv(const std::string& csv_path, const std::map<std::string, EngineResult>& results, const int frames) {
  if (csv_path.empty()) {
    return;
  }
  std::ofstream out(csv_path);
  if (!out) {
    throw std::runtime_error("Failed to open CSV output path: " + csv_path);
  }
  out << "engine,frames,time_s,fps,visited_nodes,visited_leafs,culled_nodes,accepted_leafs\n";
  for (const auto& kv : results) {
    const auto& name = kv.first;
    const auto& r = kv.second;
    const double fps = static_cast<double>(frames) / r.seconds;
    out << name << "," << frames << "," << r.seconds << "," << fps << "," << r.stats.visited_nodes << ","
        << r.stats.visited_leafs << "," << r.stats.culled_nodes << "," << r.stats.accepted_leafs << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto cfg = quakecore::ParseArgs(argc, argv);
    std::string csv_path{};
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--csv" && i + 1 < argc) {
        csv_path = argv[++i];
      }
    }

    if (cfg.threads <= 0) {
      cfg.threads = 1;
    }
    if (cfg.gpu_block_size <= 0) {
      cfg.gpu_block_size = 256;
    }

    const auto bsp = quakecore::ParseBspFile(cfg.map_path);
    const auto cameras = quakecore::GenerateCameraPath(bsp, cfg.frames, cfg.seed);

    std::map<std::string, EngineResult> results;
    results["baseline"] = RunTimed([&]() { return quakecore::RunBaselineTraversal(bsp, cameras); });
    results["cpu_opt"] = RunTimed([&]() { return quakecore::RunCpuOptimizedTraversal(bsp, cameras, cfg.threads); });
    results["gpu_opt"] = RunTimed([&]() { return quakecore::RunGpuOptimizedTraversal(bsp, cameras, cfg.gpu_block_size); });

    std::cout << "Universal benchmark results\n";
    std::cout << "map=" << cfg.map_path << " frames=" << cfg.frames << " seed=" << cfg.seed << "\n";
    const double baseline_seconds = results["baseline"].seconds;
    PrintLine("baseline", results["baseline"], cfg.frames, baseline_seconds);
    PrintLine("cpu_opt", results["cpu_opt"], cfg.frames, baseline_seconds);
    PrintLine("gpu_opt", results["gpu_opt"], cfg.frames, baseline_seconds);

    const bool cpu_ok = CheckParity(results["baseline"].stats, results["cpu_opt"].stats);
    const bool gpu_ok = CheckParity(results["baseline"].stats, results["gpu_opt"].stats);
    std::cout << "correctness_cpu_opt=" << (cpu_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "correctness_gpu_opt=" << (gpu_ok ? "PASS" : "FAIL") << "\n";

    WriteCsv(csv_path, results, cfg.frames);

    if (!cpu_ok || !gpu_ok) {
      return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "bench error: " << e.what() << "\n";
    return 1;
  }
}

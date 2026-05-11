#include "app_common.hpp"
#include "quakecore/engines.hpp"
#include "quakecore/gpu_context.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <vector>

namespace {

struct EngineResult {
  quakecore::TraversalStats stats{};
  double seconds{0.0};
  // Populated only for the gpu_amortized row; left at the sentinel below otherwise.
  double first_frame_ms{std::numeric_limits<double>::quiet_NaN()};
  double steady_median_us{std::numeric_limits<double>::quiet_NaN()};
  double steady_p99_us{std::numeric_limits<double>::quiet_NaN()};
};

template <typename Fn>
EngineResult RunTimed(Fn&& fn) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  const auto stats = fn();
  const auto t1 = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> dt = t1 - t0;
  return EngineResult{stats, dt.count(), std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN()};
}

EngineResult RunAmortized(const quakecore::BspData& bsp,
                          const std::vector<quakecore::Camera>& cameras,
                          const int frames, const int views_per_frame,
                          const int block_size) {
  EngineResult r{};
  if (frames <= 0) {
    return r;  // Caller skips the row entirely; this guards the helper.
  }

  std::vector<double> per_frame_us;
  per_frame_us.reserve(static_cast<std::size_t>(frames));

  const auto t_start = std::chrono::high_resolution_clock::now();

  quakecore::GpuTraversalContext* ctx =
      quakecore::GpuContextCreate(bsp, views_per_frame, block_size);

  std::vector<quakecore::Camera> frame_views(static_cast<std::size_t>(views_per_frame));
  quakecore::TraversalStats agg{};

  for (int f = 0; f < frames; ++f) {
    const int base = f * views_per_frame;
    for (int k = 0; k < views_per_frame; ++k) {
      frame_views[static_cast<std::size_t>(k)] =
          cameras[static_cast<std::size_t>(base + k)];
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto s = quakecore::GpuContextRun(ctx, frame_views);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::micro> us = t1 - t0;
    per_frame_us.push_back(us.count());
    agg.visited_nodes += s.visited_nodes;
    agg.visited_leafs += s.visited_leafs;
    agg.culled_nodes += s.culled_nodes;
    agg.accepted_leafs += s.accepted_leafs;
  }

  quakecore::GpuContextDestroy(ctx);

  const auto t_end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> total = t_end - t_start;

  r.stats = agg;
  r.seconds = total.count();
  r.first_frame_ms = per_frame_us.front() / 1000.0;

  if (per_frame_us.size() >= 2) {
    std::vector<double> steady(per_frame_us.begin() + 1, per_frame_us.end());
    std::sort(steady.begin(), steady.end());
    const std::size_t n = steady.size();
    r.steady_median_us = (n % 2 == 1)
                             ? steady[n / 2]
                             : 0.5 * (steady[n / 2 - 1] + steady[n / 2]);
    // 99th percentile via nearest-rank.
    std::size_t p99_idx = static_cast<std::size_t>(std::ceil(0.99 * n)) - 1;
    if (p99_idx >= n) p99_idx = n - 1;
    r.steady_p99_us = steady[p99_idx];
  }
  return r;
}

bool CheckParity(const quakecore::TraversalStats& base, const quakecore::TraversalStats& other) {
  return base.accepted_leafs == other.accepted_leafs && base.visited_nodes == other.visited_nodes;
}

void PrintLine(const std::string& name, const EngineResult& r, const int frames, const double baseline_seconds) {
  const double fps = static_cast<double>(frames) / r.seconds;
  const double speedup = baseline_seconds / r.seconds;
  std::cout << std::left << std::setw(14) << name << " "
            << "time_s=" << std::setw(10) << std::setprecision(6) << std::fixed << r.seconds << " "
            << "fps=" << std::setw(12) << std::setprecision(3) << fps << " "
            << "speedup=" << std::setw(10) << std::setprecision(3) << speedup << " "
            << "visited_nodes=" << r.stats.visited_nodes << " "
            << "accepted_leafs=" << r.stats.accepted_leafs;
  if (!std::isnan(r.first_frame_ms)) {
    std::cout << " first_frame_ms=" << std::setprecision(3) << r.first_frame_ms;
  }
  if (!std::isnan(r.steady_median_us)) {
    std::cout << " steady_median_us=" << std::setprecision(3) << r.steady_median_us;
  }
  if (!std::isnan(r.steady_p99_us)) {
    std::cout << " steady_p99_us=" << std::setprecision(3) << r.steady_p99_us;
  }
  std::cout << "\n";
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

    if (cfg.threads <= 0) cfg.threads = 1;
    if (cfg.gpu_block_size <= 0) cfg.gpu_block_size = 256;
    if (cfg.views_per_frame <= 0) cfg.views_per_frame = 1;

    const auto bsp = quakecore::ParseBspFile(cfg.map_path);
    const int total_cameras = cfg.frames * cfg.views_per_frame;
    const auto cameras = quakecore::GenerateCameraPath(bsp, total_cameras, cfg.seed);

    std::map<std::string, EngineResult> results;
    results["baseline"] = RunTimed([&]() { return quakecore::RunBaselineTraversal(bsp, cameras); });
    results["cpu_opt"] = RunTimed([&]() { return quakecore::RunCpuOptimizedTraversal(bsp, cameras, cfg.threads); });
    results["gpu_opt"] = RunTimed([&]() { return quakecore::RunGpuOptimizedTraversal(bsp, cameras, cfg.gpu_block_size); });
    if (cfg.frames > 0) {
      results["gpu_amortized"] =
          RunAmortized(bsp, cameras, cfg.frames, cfg.views_per_frame, cfg.gpu_block_size);
    }

    std::cout << "Universal benchmark results\n";
    std::cout << "map=" << cfg.map_path
              << " frames=" << cfg.frames
              << " views_per_frame=" << cfg.views_per_frame
              << " seed=" << cfg.seed << "\n";

    const double baseline_seconds = results["baseline"].seconds;
    // Pass total_cameras as the "frames" argument everywhere so fps is "cameras per
    // second" uniformly across rows.
    PrintLine("baseline", results["baseline"], total_cameras, baseline_seconds);
    PrintLine("cpu_opt", results["cpu_opt"], total_cameras, baseline_seconds);
    PrintLine("gpu_opt", results["gpu_opt"], total_cameras, baseline_seconds);
    if (cfg.frames > 0) {
      PrintLine("gpu_amortized", results["gpu_amortized"], total_cameras, baseline_seconds);
    }

    const bool cpu_ok = CheckParity(results["baseline"].stats, results["cpu_opt"].stats);
    const bool gpu_ok = CheckParity(results["baseline"].stats, results["gpu_opt"].stats);
    const bool gpu_am_ok = (cfg.frames == 0) ? true
                          : CheckParity(results["baseline"].stats,
                                        results["gpu_amortized"].stats);
    std::cout << "correctness_cpu_opt=" << (cpu_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "correctness_gpu_opt=" << (gpu_ok ? "PASS" : "FAIL") << "\n";
    if (cfg.frames > 0) {
      std::cout << "correctness_gpu_amortized=" << (gpu_am_ok ? "PASS" : "FAIL") << "\n";
    }

    WriteCsv(csv_path, results, total_cameras);

    if (!cpu_ok || !gpu_ok || !gpu_am_ok) {
      return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "bench error: " << e.what() << "\n";
    return 1;
  }
}

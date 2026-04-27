#include "app_common.hpp"
#include "quakecore/engines.hpp"

int main(int argc, char** argv) {
  try {
    const auto cfg = quakecore::ParseArgs(argc, argv);
    const auto bsp = quakecore::ParseBspFile(cfg.map_path);
    const auto cameras = quakecore::GenerateCameraPath(bsp, cfg.frames, cfg.seed);

    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto stats = quakecore::RunBaselineTraversal(bsp, cameras);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> dt = t1 - t0;
    quakecore::PrintSummary("baseline", stats, dt.count(), cfg.frames);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "baseline error: " << e.what() << "\n";
    return 1;
  }
}

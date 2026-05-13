#include "frame_runner.hpp"
#include "quakecore/engines.hpp"
#include <chrono>
#include <vector>

namespace quakecore::live {

static Camera CameraFromPacket(const QcfpFramePacket& p) {
  Camera c{};
  c.position = {p.cam_origin[0], p.cam_origin[1], p.cam_origin[2]};
  c.forward  = {p.cam_forward[0], p.cam_forward[1], p.cam_forward[2]};
  c.right    = {p.cam_right[0],   p.cam_right[1],   p.cam_right[2]};
  c.up       = {p.cam_up[0],      p.cam_up[1],      p.cam_up[2]};
  return c;
}

static Frustum FrustumFromPacket(const QcfpFramePacket& p) {
  Frustum f{};
  for (int i = 0; i < 6; ++i) {
    f.planes[i].normal = {p.planes[i].n[0], p.planes[i].n[1], p.planes[i].n[2]};
    f.planes[i].dist   = p.planes[i].d;
    f.planes[i].type   = 3;
  }
  return f;
}

template <typename Fn>
static uint64_t TimeNs(Fn&& fn) {
  auto t0 = std::chrono::steady_clock::now();
  fn();
  auto t1 = std::chrono::steady_clock::now();
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

FrameResult RunOneFrame(const BspData& bsp, const QcfpFramePacket& pkt,
                        int threads, int gpu_block_size) {
  std::vector<Camera> cams{ CameraFromPacket(pkt) };
  Frustum frustum = FrustumFromPacket(pkt);

  TraversalStats s_base{}, s_cpu{}, s_gpu{};
  uint64_t t_base = TimeNs([&]{ s_base = RunBaselineTraversal(bsp, cams, &frustum); });
  uint64_t t_cpu  = TimeNs([&]{ s_cpu  = RunCpuOptimizedTraversal(bsp, cams, threads, &frustum); });
  uint64_t t_gpu  = TimeNs([&]{ s_gpu  = RunGpuOptimizedTraversal(bsp, cams, gpu_block_size, &frustum); });

  bool ok = (s_base.visited_nodes == s_cpu.visited_nodes)
         && (s_base.accepted_leafs == s_cpu.accepted_leafs)
         && (s_base.visited_nodes == s_gpu.visited_nodes)
         && (s_base.accepted_leafs == s_gpu.accepted_leafs);

  return FrameResult{
    pkt.frame_id,
    pkt.t_game_cull_end_ns - pkt.t_game_cull_start_ns,
    t_base, t_cpu, t_gpu,
    pkt.game_visited_nodes, pkt.game_accepted_leafs,
    s_base.visited_nodes, s_base.accepted_leafs,
    ok
  };
}

}  // namespace quakecore::live

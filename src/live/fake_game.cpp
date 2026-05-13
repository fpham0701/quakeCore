#include "quakecore/bsp_parser.hpp"
#include "quakecore/camera_path.hpp"
#include "quakecore/frustum.hpp"
#include "quakecore_live/frame_protocol.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  std::string transport, map; int frames = 256; uint64_t seed = 7;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if      (s == "--transport") transport = argv[++i];
    else if (s == "--map")       map = argv[++i];
    else if (s == "--frames")    frames = std::atoi(argv[++i]);
    else if (s == "--seed")      seed = std::strtoull(argv[++i], nullptr, 10);
  }
  QcfpTransport* t = qcfp_open_producer(transport.c_str());
  if (!t) { std::fprintf(stderr, "open producer failed\n"); return 1; }

  uint8_t sha[32]; qcfp_hash_file(map.c_str(), sha);
  std::vector<uint8_t> hs_buf(sizeof(QcfpHandshakePacket) + map.size());
  auto* hs = reinterpret_cast<QcfpHandshakePacket*>(hs_buf.data());
  hs->hdr.magic = QCFP_MAGIC; hs->hdr.version = QCFP_VERSION;
  hs->hdr.type = QCFP_TYPE_HANDSHAKE;
  hs->hdr.length = (uint32_t)(hs_buf.size() - sizeof(QcfpHeader));
  std::memcpy(hs->bsp_sha256, sha, 32);
  hs->bsp_path_len = (uint16_t)map.size();
  std::memcpy(hs_buf.data() + sizeof(QcfpHandshakePacket), map.data(), map.size());
  if (qcfp_send(t, hs_buf.data(), hs_buf.size()) != QCFP_OK) { std::fprintf(stderr, "handshake send failed\n"); return 1; }

  auto bsp = quakecore::ParseBspFile(map);
  auto cams = quakecore::GenerateCameraPath(bsp, frames, seed);

  for (int i = 0; i < (int)cams.size(); ++i) {
    quakecore::Frustum f = quakecore::BuildFrustum(cams[i]);
    QcfpFramePacket p{};
    p.hdr.magic = QCFP_MAGIC; p.hdr.version = QCFP_VERSION;
    p.hdr.type = QCFP_TYPE_FRAME;
    p.hdr.length = sizeof(p) - sizeof(QcfpHeader);
    p.frame_id = (uint64_t)i;
    p.t_game_cull_start_ns = 0; p.t_game_cull_end_ns = 0;
    std::memcpy(p.bsp_sha256, sha, 32);
    p.cam_origin[0]=cams[i].position.x; p.cam_origin[1]=cams[i].position.y; p.cam_origin[2]=cams[i].position.z;
    p.cam_forward[0]=cams[i].forward.x; p.cam_forward[1]=cams[i].forward.y; p.cam_forward[2]=cams[i].forward.z;
    p.cam_right[0]=cams[i].right.x;     p.cam_right[1]=cams[i].right.y;     p.cam_right[2]=cams[i].right.z;
    p.cam_up[0]=cams[i].up.x;           p.cam_up[1]=cams[i].up.y;           p.cam_up[2]=cams[i].up.z;
    p.n_planes = 6;
    for (int k = 0; k < 6; ++k) {
      p.planes[k].n[0] = f.planes[k].normal.x;
      p.planes[k].n[1] = f.planes[k].normal.y;
      p.planes[k].n[2] = f.planes[k].normal.z;
      p.planes[k].d    = f.planes[k].dist;
    }
    QcfpStatus s = qcfp_send(t, &p, sizeof p);
    if (s == QCFP_ERROR) { std::fprintf(stderr, "send error\n"); return 1; }
    while (s == QCFP_DROPPED) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      s = qcfp_send(t, &p, sizeof p);
    }
  }
  qcfp_close(t);
  return 0;
}

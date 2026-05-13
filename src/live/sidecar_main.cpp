#include "frame_runner.hpp"
#include "quakecore/bsp_parser.hpp"
#include "quakecore_live/frame_protocol.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace qc = quakecore;
namespace qcl = quakecore::live;

struct Args {
  std::string transport;
  std::string csv;
  std::string record;
  std::string replay;
  int threads = 8;
  int block_size = 256;
};

static Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if      (s == "--transport") a.transport = next();
    else if (s == "--csv")       a.csv = next();
    else if (s == "--record")    a.record = next();
    else if (s == "--replay")    a.replay = next();
    else if (s == "--threads")   a.threads = std::atoi(next().c_str());
    else if (s == "--block-size")a.block_size = std::atoi(next().c_str());
    else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); std::exit(64); }
  }
  if (a.transport.empty() && a.replay.empty()) {
    std::fprintf(stderr, "need --transport or --replay\n"); std::exit(64);
  }
  return a;
}

static void WriteCsvHeader(std::ostream& out) {
  out << "frame_id,game_ns,baseline_ns,cpu_ns,gpu_ns,"
         "game_visited,game_accepted,qc_visited,qc_accepted,drops_so_far\n";
}
static void WriteCsvRow(std::ostream& out, const qcl::FrameResult& r, uint64_t drops) {
  out << r.frame_id << ',' << r.game_ns << ',' << r.baseline_ns << ',' << r.cpu_ns << ','
      << r.gpu_ns << ',' << r.game_visited_nodes << ',' << r.game_accepted_leafs << ','
      << r.qc_visited_nodes << ',' << r.qc_accepted_leafs << ',' << drops << '\n';
}

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);

  std::ofstream csv;
  if (!args.csv.empty()) { csv.open(args.csv); if (!csv) { std::perror("csv"); return 1; } WriteCsvHeader(csv); }
  std::ofstream record_bin;
  if (!args.record.empty()) record_bin.open(args.record, std::ios::binary);

  auto run_with_transport = [&](QcfpTransport* t) -> int {
    std::vector<uint8_t> buf(sizeof(QcfpHandshakePacket) + 4096);
    size_t got = 0;
    if (qcfp_recv(t, buf.data(), buf.size(), &got) != QCFP_OK) {
      std::fprintf(stderr, "no handshake\n"); return 1;
    }
    if (record_bin) { uint32_t l = (uint32_t)got; record_bin.write((char*)&l, 4); record_bin.write((char*)buf.data(), got); }
    auto* hs = reinterpret_cast<QcfpHandshakePacket*>(buf.data());
    if (hs->hdr.magic != QCFP_MAGIC || hs->hdr.type != QCFP_TYPE_HANDSHAKE) {
      std::fprintf(stderr, "bad handshake\n"); return 1;
    }
    std::string bsp_path((const char*)(buf.data() + sizeof(QcfpHandshakePacket)), hs->bsp_path_len);
    uint8_t actual[32];
    if (qcfp_hash_file(bsp_path.c_str(), actual) != 0) { std::fprintf(stderr, "cannot hash %s\n", bsp_path.c_str()); return 1; }
    if (std::memcmp(actual, hs->bsp_sha256, 32) != 0) {
      std::fprintf(stderr, "BSP sha256 mismatch for %s\n", bsp_path.c_str()); return 1;
    }

    qc::BspData bsp = qc::ParseBspFile(bsp_path);

    QcfpFramePacket pkt;
    for (;;) {
      QcfpStatus s = qcfp_recv(t, &pkt, sizeof pkt, &got);
      if (s == QCFP_EOF) break;
      if (s != QCFP_OK) { std::fprintf(stderr, "recv error %d\n", s); return 1; }
      if (record_bin) { uint32_t l = (uint32_t)got; record_bin.write((char*)&l, 4); record_bin.write((char*)&pkt, got); }
      if (pkt.hdr.magic != QCFP_MAGIC || pkt.hdr.type != QCFP_TYPE_FRAME) continue;

      qcl::FrameResult r = qcl::RunOneFrame(bsp, pkt, args.threads, args.block_size);
      if (!r.internal_parity_ok) {
        std::fprintf(stderr, "INTERNAL PARITY FAIL at frame %llu\n", (unsigned long long)r.frame_id);
        return 2;
      }
      if (csv) WriteCsvRow(csv, r, qcfp_dropped(t));
    }
    return 0;
  };

  if (!args.replay.empty()) {
    std::ifstream in(args.replay, std::ios::binary);
    if (!in) { std::perror("replay"); return 1; }
    auto read_one = [&](std::vector<uint8_t>& buf) -> bool {
      uint32_t l = 0; in.read((char*)&l, 4); if (!in) return false;
      buf.resize(l); in.read((char*)buf.data(), l); return (bool)in;
    };
    std::vector<uint8_t> hs_buf;
    if (!read_one(hs_buf)) { std::fprintf(stderr, "replay: no handshake\n"); return 1; }
    auto* hs = reinterpret_cast<QcfpHandshakePacket*>(hs_buf.data());
    std::string bsp_path((const char*)(hs_buf.data() + sizeof(QcfpHandshakePacket)), hs->bsp_path_len);
    qc::BspData bsp = qc::ParseBspFile(bsp_path);
    std::vector<uint8_t> fbuf;
    while (read_one(fbuf)) {
      if (fbuf.size() < sizeof(QcfpFramePacket)) continue;
      QcfpFramePacket p; std::memcpy(&p, fbuf.data(), sizeof p);
      qcl::FrameResult r = qcl::RunOneFrame(bsp, p, args.threads, args.block_size);
      if (!r.internal_parity_ok) { std::fprintf(stderr, "PARITY FAIL frame %llu\n", (unsigned long long)r.frame_id); return 2; }
      if (csv) WriteCsvRow(csv, r, 0);
    }
    return 0;
  }

  QcfpTransport* t = qcfp_open_consumer(args.transport.c_str());
  if (!t) { std::fprintf(stderr, "cannot open transport %s\n", args.transport.c_str()); return 1; }
  int rc = run_with_transport(t);
  qcfp_close(t);
  return rc;
}

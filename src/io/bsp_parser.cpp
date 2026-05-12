#include "quakecore/bsp_parser.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace quakecore {
namespace {

constexpr int32_t kBspV29 = 29;
constexpr int32_t kBspV2  = static_cast<int32_t>('B' | ('S' << 8) | ('P' << 16) | ('2' << 24));
constexpr int32_t kBsp2rmq = static_cast<int32_t>(('B' << 24) | ('S' << 16) | ('P' << 8) | '2');
constexpr int32_t kBspHl   = 30;

constexpr int kHeaderLumps = 15;
constexpr int kLumpPlanes = 1;
constexpr int kLumpVertices = 3;
constexpr int kLumpNodes = 5;
constexpr int kLumpLeafs = 10;
constexpr int kLumpModels = 14;

struct Lump {
  int32_t fileofs;
  int32_t filelen;
};

struct Header {
  int32_t version;
  Lump lumps[kHeaderLumps];
};

template <typename T>
std::vector<T> ReadLumpAs(const std::vector<uint8_t>& bytes, const Lump& lump, const char* name) {
  if (lump.fileofs < 0 || lump.filelen < 0) {
    throw std::runtime_error(std::string("Invalid lump bounds: ") + name);
  }
  const size_t offset = static_cast<size_t>(lump.fileofs);
  const size_t length = static_cast<size_t>(lump.filelen);
  if (offset + length > bytes.size()) {
    throw std::runtime_error(std::string("Lump exceeds file size: ") + name);
  }
  if ((length % sizeof(T)) != 0U) {
    throw std::runtime_error(std::string("Lump has invalid element size: ") + name);
  }
  const size_t count = length / sizeof(T);
  std::vector<T> out(count);
  if (count > 0) {
    std::memcpy(out.data(), bytes.data() + offset, length);
  }
  return out;
}

// On-disk shared lumps (identical in v29 and BSP2).
struct DVertex {
  float point[3];
};
struct DPlane {
  float normal[3];
  float dist;
  int32_t type;
};
struct DModel {
  float mins[3];
  float maxs[3];
  float origin[3];
  int32_t headnode[4];
  int32_t visleafs;
  int32_t firstface;
  int32_t numfaces;
};

// v29 on-disk node / leaf — 24 / 28 bytes. From legacy_src/bspfile.h.
struct DNodeV29 {
  int32_t planenum;
  int16_t children[2];
  int16_t mins[3];
  int16_t maxs[3];
  uint16_t firstface;
  uint16_t numfaces;
};
static_assert(sizeof(DNodeV29) == 24, "DNodeV29 must be 24 bytes");

struct DLeafV29 {
  int32_t contents;
  int32_t visofs;
  int16_t mins[3];
  int16_t maxs[3];
  uint16_t firstmarksurface;
  uint16_t nummarksurfaces;
  uint8_t  ambient_level[4];
};
static_assert(sizeof(DLeafV29) == 28, "DLeafV29 must be 28 bytes");

// BSP2 on-disk node / leaf — 44 / 44 bytes.
// From ericw-tools/include/common/bspfile_q1.hh at tag v0.18.1:
//   struct bsp2_dnode_t  { int32 planenum; int32 children[2]; float mins[3]; float maxs[3]; uint32 firstface; uint32 numfaces; }
//   struct bsp2_dleaf_t  { int32 contents; int32 visofs; float mins[3]; float maxs[3]; uint32 firstmarksurface; uint32 nummarksurfaces; uint8 ambient_level[4]; }
struct DNodeV2 {
  int32_t planenum;
  int32_t children[2];
  float   mins[3];
  float   maxs[3];
  uint32_t firstface;
  uint32_t numfaces;
};
static_assert(sizeof(DNodeV2) == 44, "DNodeV2 must be 44 bytes");

struct DLeafV2 {
  int32_t contents;
  int32_t visofs;
  float   mins[3];
  float   maxs[3];
  uint32_t firstmarksurface;
  uint32_t nummarksurfaces;
  uint8_t  ambient_level[4];
};
static_assert(sizeof(DLeafV2) == 44, "DLeafV2 must be 44 bytes");

template <typename DNodeT>
void AppendNode(std::vector<BspNodeDisk>& out, const DNodeT& n) {
  BspNodeDisk node{};
  node.planenum    = n.planenum;
  node.children[0] = static_cast<int32_t>(n.children[0]);
  node.children[1] = static_cast<int32_t>(n.children[1]);
  node.mins[0]     = static_cast<float>(n.mins[0]);
  node.mins[1]     = static_cast<float>(n.mins[1]);
  node.mins[2]     = static_cast<float>(n.mins[2]);
  node.maxs[0]     = static_cast<float>(n.maxs[0]);
  node.maxs[1]     = static_cast<float>(n.maxs[1]);
  node.maxs[2]     = static_cast<float>(n.maxs[2]);
  node.firstface   = static_cast<int32_t>(n.firstface);
  node.numfaces    = static_cast<int32_t>(n.numfaces);
  out.push_back(node);
}

template <typename DLeafT>
void AppendLeaf(std::vector<BspLeafDisk>& out, const DLeafT& l) {
  BspLeafDisk leaf{};
  leaf.contents         = l.contents;
  leaf.visofs           = l.visofs;
  leaf.mins[0]          = static_cast<float>(l.mins[0]);
  leaf.mins[1]          = static_cast<float>(l.mins[1]);
  leaf.mins[2]          = static_cast<float>(l.mins[2]);
  leaf.maxs[0]          = static_cast<float>(l.maxs[0]);
  leaf.maxs[1]          = static_cast<float>(l.maxs[1]);
  leaf.maxs[2]          = static_cast<float>(l.maxs[2]);
  leaf.firstmarksurface = static_cast<int32_t>(l.firstmarksurface);
  leaf.nummarksurfaces  = static_cast<int32_t>(l.nummarksurfaces);
  out.push_back(leaf);
}

}  // namespace

BspData ParseBspFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("Failed to open BSP file: " + path);
  }
  const std::streamsize file_size = in.tellg();
  if (file_size < static_cast<std::streamsize>(sizeof(Header))) {
    throw std::runtime_error("BSP file too small: " + path);
  }
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
  if (!in.read(reinterpret_cast<char*>(bytes.data()), file_size)) {
    throw std::runtime_error("Failed to read BSP file: " + path);
  }

  Header header{};
  std::memcpy(&header, bytes.data(), sizeof(Header));

  const bool is_v29 = (header.version == kBspV29);
  const bool is_v2  = (header.version == kBspV2);
  if (!is_v29 && !is_v2) {
    std::ostringstream oss;
    oss << "Unsupported BSP magic: 0x" << std::hex << header.version
        << ". Supported: v29 (0x1D), BSP2 (0x32505342).";
    if (header.version == kBsp2rmq) {
      oss << " (Got BSP29a/2PSB — intermediate format, not supported.)";
    } else if (header.version == kBspHl) {
      oss << " (Got Half-Life BSP v30 — not supported.)";
    }
    throw std::runtime_error(oss.str());
  }

  const auto dverts  = ReadLumpAs<DVertex>(bytes, header.lumps[kLumpVertices], "vertexes");
  const auto dplanes = ReadLumpAs<DPlane>(bytes, header.lumps[kLumpPlanes], "planes");
  const auto dmodels = ReadLumpAs<DModel>(bytes, header.lumps[kLumpModels], "models");

  BspData bsp{};
  bsp.source_path = path;

  bsp.vertices.reserve(dverts.size());
  for (const auto& v : dverts) {
    bsp.vertices.push_back(Vec3{v.point[0], v.point[1], v.point[2]});
  }

  bsp.planes.reserve(dplanes.size());
  for (const auto& p : dplanes) {
    bsp.planes.push_back(Plane{Vec3{p.normal[0], p.normal[1], p.normal[2]}, p.dist, p.type});
  }

  if (is_v29) {
    const auto dnodes = ReadLumpAs<DNodeV29>(bytes, header.lumps[kLumpNodes], "nodes (v29)");
    const auto dleafs = ReadLumpAs<DLeafV29>(bytes, header.lumps[kLumpLeafs], "leafs (v29)");
    bsp.nodes.reserve(dnodes.size());
    for (const auto& n : dnodes) AppendNode(bsp.nodes, n);
    bsp.leafs.reserve(dleafs.size());
    for (const auto& l : dleafs) AppendLeaf(bsp.leafs, l);
  } else {
    const auto dnodes = ReadLumpAs<DNodeV2>(bytes, header.lumps[kLumpNodes], "nodes (BSP2)");
    const auto dleafs = ReadLumpAs<DLeafV2>(bytes, header.lumps[kLumpLeafs], "leafs (BSP2)");
    bsp.nodes.reserve(dnodes.size());
    for (const auto& n : dnodes) AppendNode(bsp.nodes, n);
    bsp.leafs.reserve(dleafs.size());
    for (const auto& l : dleafs) AppendLeaf(bsp.leafs, l);
  }

  bsp.models.reserve(dmodels.size());
  for (const auto& m : dmodels) {
    BspModel model{};
    model.mins = Vec3{m.mins[0], m.mins[1], m.mins[2]};
    model.maxs = Vec3{m.maxs[0], m.maxs[1], m.maxs[2]};
    for (int i = 0; i < 4; ++i) {
      model.headnode[i] = m.headnode[i];
    }
    model.visleafs = m.visleafs;
    bsp.models.push_back(model);
  }

  if (bsp.models.empty()) {
    throw std::runtime_error("BSP contains no models");
  }
  if (bsp.nodes.empty()) {
    throw std::runtime_error("BSP contains no nodes");
  }

  return bsp;
}

}  // namespace quakecore

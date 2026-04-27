#include "quakecore/bsp_parser.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace quakecore {
namespace {

constexpr int kBspVersion = 29;
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
  if (header.version != kBspVersion) {
    throw std::runtime_error("Unsupported BSP version: " + std::to_string(header.version));
  }

  struct DVertex {
    float point[3];
  };
  struct DPlane {
    float normal[3];
    float dist;
    int32_t type;
  };
  struct DNode {
    int32_t planenum;
    int16_t children[2];
    int16_t mins[3];
    int16_t maxs[3];
    uint16_t firstface;
    uint16_t numfaces;
  };
  struct DLeaf {
    int32_t contents;
    int32_t visofs;
    int16_t mins[3];
    int16_t maxs[3];
    uint16_t firstmarksurface;
    uint16_t nummarksurfaces;
    uint8_t ambient_level[4];
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

  const auto dverts = ReadLumpAs<DVertex>(bytes, header.lumps[kLumpVertices], "vertexes");
  const auto dplanes = ReadLumpAs<DPlane>(bytes, header.lumps[kLumpPlanes], "planes");
  const auto dnodes = ReadLumpAs<DNode>(bytes, header.lumps[kLumpNodes], "nodes");
  const auto dleafs = ReadLumpAs<DLeaf>(bytes, header.lumps[kLumpLeafs], "leafs");
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

  bsp.nodes.reserve(dnodes.size());
  for (const auto& n : dnodes) {
    BspNodeDisk node{};
    node.planenum = n.planenum;
    node.children[0] = n.children[0];
    node.children[1] = n.children[1];
    node.mins[0] = n.mins[0];
    node.mins[1] = n.mins[1];
    node.mins[2] = n.mins[2];
    node.maxs[0] = n.maxs[0];
    node.maxs[1] = n.maxs[1];
    node.maxs[2] = n.maxs[2];
    bsp.nodes.push_back(node);
  }

  bsp.leafs.reserve(dleafs.size());
  for (const auto& l : dleafs) {
    BspLeafDisk leaf{};
    leaf.contents = l.contents;
    leaf.visofs = l.visofs;
    leaf.mins[0] = l.mins[0];
    leaf.mins[1] = l.mins[1];
    leaf.mins[2] = l.mins[2];
    leaf.maxs[0] = l.maxs[0];
    leaf.maxs[1] = l.maxs[1];
    leaf.maxs[2] = l.maxs[2];
    bsp.leafs.push_back(leaf);
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

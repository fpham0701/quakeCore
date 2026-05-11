#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace quakecore {

struct Vec3 {
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
};

struct Plane {
  Vec3 normal{};
  float dist{0.0F};
  int32_t type{0};
};

struct BspNodeDisk {
  int32_t planenum{0};
  int16_t children[2]{0, 0};  // <0 encodes leaf index as -(leaf+1)
  int16_t mins[3]{0, 0, 0};
  int16_t maxs[3]{0, 0, 0};
};

struct BspLeafDisk {
  int32_t contents{0};
  int32_t visofs{0};
  int16_t mins[3]{0, 0, 0};
  int16_t maxs[3]{0, 0, 0};
};

struct BspModel {
  Vec3 mins{};
  Vec3 maxs{};
  int32_t headnode[4]{0, 0, 0, 0};
  int32_t visleafs{0};
};

struct BspData {
  std::string source_path{};
  std::vector<Vec3> vertices{};
  std::vector<Plane> planes{};
  std::vector<BspNodeDisk> nodes{};
  std::vector<BspLeafDisk> leafs{};
  std::vector<BspModel> models{};
};

struct Frustum {
  std::array<Plane, 6> planes{};
};

struct Camera {
  Vec3 position{};
  Vec3 forward{1.0F, 0.0F, 0.0F};
  Vec3 right{0.0F, 1.0F, 0.0F};
  Vec3 up{0.0F, 0.0F, 1.0F};
  float near_plane{0.1F};
  float far_plane{4096.0F};
  float fov_y_radians{1.0471975512F};
  float aspect_ratio{16.0F / 9.0F};
};

struct TraversalStats {
  uint64_t visited_nodes{0};
  uint64_t visited_leafs{0};
  uint64_t culled_nodes{0};
  uint64_t accepted_leafs{0};
};

struct BenchmarkConfig {
  std::string map_path{};
  int frames{1024};
  uint64_t seed{12345};
  int threads{1};
  int gpu_block_size{256};
  int views_per_frame{1};
};

}  // namespace quakecore

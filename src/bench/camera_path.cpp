#include "quakecore/camera_path.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace quakecore {

std::vector<Camera> GenerateCameraPath(const BspData& bsp, const int frames, const uint64_t seed) {
  std::vector<Camera> out;
  if (frames <= 0 || bsp.models.empty()) {
    return out;
  }

  const auto& world = bsp.models.front();
  const Vec3 center{
      0.5F * (world.mins.x + world.maxs.x),
      0.5F * (world.mins.y + world.maxs.y),
      0.5F * (world.mins.z + world.maxs.z),
  };
  const float radius_x = std::max(64.0F, (world.maxs.x - world.mins.x) * 0.35F);
  const float radius_y = std::max(64.0F, (world.maxs.y - world.mins.y) * 0.35F);
  const float z_span = std::max(32.0F, (world.maxs.z - world.mins.z) * 0.15F);

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> jitter(-0.15F, 0.15F);

  out.reserve(static_cast<size_t>(frames));
  for (int i = 0; i < frames; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(frames);
    const float theta = 2.0F * 3.1415926535F * (2.0F * t + 0.1F * jitter(rng));

    Camera cam{};
    cam.position.x = center.x + radius_x * std::cos(theta);
    cam.position.y = center.y + radius_y * std::sin(theta);
    cam.position.z = center.z + z_span * std::sin(theta * 0.5F);

    Vec3 look = Vec3{center.x - cam.position.x, center.y - cam.position.y, center.z - cam.position.z};
    const float inv_len = 1.0F / std::sqrt(std::max(look.x * look.x + look.y * look.y + look.z * look.z, 1.0e-12F));
    look.x *= inv_len;
    look.y *= inv_len;
    look.z *= inv_len;
    cam.forward = look;
    cam.right = Vec3{-look.y, look.x, 0.0F};
    const float inv_r = 1.0F / std::sqrt(std::max(cam.right.x * cam.right.x + cam.right.y * cam.right.y, 1.0e-12F));
    cam.right.x *= inv_r;
    cam.right.y *= inv_r;
    cam.up = Vec3{
        cam.right.y * cam.forward.z - cam.right.z * cam.forward.y,
        cam.right.z * cam.forward.x - cam.right.x * cam.forward.z,
        cam.right.x * cam.forward.y - cam.right.y * cam.forward.x,
    };
    out.push_back(cam);
  }
  return out;
}

}  // namespace quakecore

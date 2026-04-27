#include "quakecore/frustum.hpp"

#include <algorithm>
#include <cmath>

namespace quakecore {
namespace {

inline Vec3 Add(const Vec3& a, const Vec3& b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 Sub(const Vec3& a, const Vec3& b) { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 Mul(const Vec3& v, const float s) { return Vec3{v.x * s, v.y * s, v.z * s}; }
inline float Dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 Cross(const Vec3& a, const Vec3& b) {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 Normalize(const Vec3& v) {
  const float len = std::sqrt(std::max(Dot(v, v), 1.0e-20F));
  return Vec3{v.x / len, v.y / len, v.z / len};
}

Plane MakePlaneFromPointNormal(const Vec3& point, const Vec3& normal) {
  const Vec3 n = Normalize(normal);
  return Plane{n, Dot(n, point), 0};
}

}  // namespace

Frustum BuildFrustum(const Camera& camera) {
  const Vec3 f = Normalize(camera.forward);
  const Vec3 r = Normalize(camera.right);
  const Vec3 u = Normalize(camera.up);

  const float tan_half_fovy = std::tan(camera.fov_y_radians * 0.5F);
  const float near_h = camera.near_plane * tan_half_fovy;
  const float near_w = near_h * camera.aspect_ratio;
  const float far_h = camera.far_plane * tan_half_fovy;
  const float far_w = far_h * camera.aspect_ratio;

  const Vec3 nc = Add(camera.position, Mul(f, camera.near_plane));
  const Vec3 fc = Add(camera.position, Mul(f, camera.far_plane));

  const Vec3 ntl = Add(Add(nc, Mul(u, near_h)), Mul(r, -near_w));
  const Vec3 ntr = Add(Add(nc, Mul(u, near_h)), Mul(r, near_w));
  const Vec3 nbl = Add(Add(nc, Mul(u, -near_h)), Mul(r, -near_w));
  const Vec3 nbr = Add(Add(nc, Mul(u, -near_h)), Mul(r, near_w));
  const Vec3 ftl = Add(Add(fc, Mul(u, far_h)), Mul(r, -far_w));
  const Vec3 ftr = Add(Add(fc, Mul(u, far_h)), Mul(r, far_w));
  const Vec3 fbl = Add(Add(fc, Mul(u, -far_h)), Mul(r, -far_w));
  const Vec3 fbr = Add(Add(fc, Mul(u, -far_h)), Mul(r, far_w));

  Frustum fr{};
  fr.planes[0] = MakePlaneFromPointNormal(nc, f);                    // near
  fr.planes[1] = MakePlaneFromPointNormal(fc, Mul(f, -1.0F));        // far
  fr.planes[2] = MakePlaneFromPointNormal(camera.position, Cross(nbl, ntl));  // left
  fr.planes[3] = MakePlaneFromPointNormal(camera.position, Cross(ntr, nbr));  // right
  fr.planes[4] = MakePlaneFromPointNormal(camera.position, Cross(ntl, ntr));  // top
  fr.planes[5] = MakePlaneFromPointNormal(camera.position, Cross(nbr, nbl));  // bottom

  (void)ftl;
  (void)ftr;
  (void)fbl;
  (void)fbr;
  return fr;
}

bool AabbIntersectsFrustum(const Frustum& frustum, const Vec3& mins, const Vec3& maxs) {
  for (const auto& p : frustum.planes) {
    Vec3 reject{};
    reject.x = (p.normal.x >= 0.0F) ? mins.x : maxs.x;
    reject.y = (p.normal.y >= 0.0F) ? mins.y : maxs.y;
    reject.z = (p.normal.z >= 0.0F) ? mins.z : maxs.z;
    if (Dot(p.normal, reject) - p.dist > 0.0F) {
      return false;
    }
  }
  return true;
}

}  // namespace quakecore

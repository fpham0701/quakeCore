#pragma once

#include "quakecore/types.hpp"

namespace quakecore {

/**
 * Build a perspective frustum from camera vectors and clip distances.
 */
Frustum BuildFrustum(const Camera& camera);

/**
 * Test an AABB against a frustum. Returns true if box is visible or intersecting.
 */
bool AabbIntersectsFrustum(const Frustum& frustum, const Vec3& mins, const Vec3& maxs);

}  // namespace quakecore

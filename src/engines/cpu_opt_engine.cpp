#include "quakecore/engines.hpp"

#include <array>
#include <cstddef>
#include <immintrin.h>
#include <stdexcept>
#include <vector>

#include "quakecore/frustum.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace quakecore {
namespace {

/**
 * 32-byte aligned SoA storage; required by AVX2 load/store intrinsics.
 */
struct alignas(32) NodeSoA {
  std::vector<float> minx, miny, minz;
  std::vector<float> maxx, maxy, maxz;
  std::vector<int32_t> child0, child1;
};

NodeSoA BuildNodeSoA(const BspData& bsp) {
  NodeSoA soa{};
  const size_t n = bsp.nodes.size();
  soa.minx.resize(n);
  soa.miny.resize(n);
  soa.minz.resize(n);
  soa.maxx.resize(n);
  soa.maxy.resize(n);
  soa.maxz.resize(n);
  soa.child0.resize(n);
  soa.child1.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const auto& node = bsp.nodes[i];
    soa.minx[i] = static_cast<float>(node.mins[0]);
    soa.miny[i] = static_cast<float>(node.mins[1]);
    soa.minz[i] = static_cast<float>(node.mins[2]);
    soa.maxx[i] = static_cast<float>(node.maxs[0]);
    soa.maxy[i] = static_cast<float>(node.maxs[1]);
    soa.maxz[i] = static_cast<float>(node.maxs[2]);
    soa.child0[i] = node.children[0];
    soa.child1[i] = node.children[1];
  }
  return soa;
}

bool IntersectsPlaneAvx(const Plane& p, const std::array<float, 6>& box) {
  alignas(32) float vals[8]{};
  vals[0] = (p.normal.x >= 0.0F) ? box[0] : box[3];
  vals[1] = (p.normal.y >= 0.0F) ? box[1] : box[4];
  vals[2] = (p.normal.z >= 0.0F) ? box[2] : box[5];
  vals[3] = 1.0F;
  vals[4] = p.normal.x;
  vals[5] = p.normal.y;
  vals[6] = p.normal.z;
  vals[7] = -p.dist;
  const __m256 v = _mm256_load_ps(vals);
  const __m256 mul = _mm256_mul_ps(v, _mm256_permute2f128_ps(v, v, 0x01));
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, mul);
  const float distance = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  return distance <= 0.0F;
}

bool AabbIntersectsFrustumAvx(const Frustum& frustum, const std::array<float, 6>& box) {
  for (const auto& p : frustum.planes) {
    if (!IntersectsPlaneAvx(p, box)) {
      return false;
    }
  }
  return true;
}

TraversalStats TraverseCamera(const BspData& bsp, const NodeSoA& soa, const Camera& cam) {
  TraversalStats stats{};
  const Frustum fr = BuildFrustum(cam);
  std::vector<int32_t> stack;
  stack.push_back(bsp.models.front().headnode[0]);

  while (!stack.empty()) {
    const int32_t idx = stack.back();
    stack.pop_back();
    if (idx < 0 || idx >= static_cast<int32_t>(soa.child0.size())) {
      continue;
    }
    stats.visited_nodes++;
    const std::array<float, 6> box{soa.minx[idx], soa.miny[idx], soa.minz[idx], soa.maxx[idx], soa.maxy[idx], soa.maxz[idx]};
    if (!AabbIntersectsFrustumAvx(fr, box)) {
      stats.culled_nodes++;
      continue;
    }
    const int32_t c0 = soa.child0[idx];
    const int32_t c1 = soa.child1[idx];
    for (const int32_t c : {c0, c1}) {
      if (c >= 0) {
        stack.push_back(c);
      } else {
        const int32_t leaf_idx = -1 - c;
        if (leaf_idx >= 0 && leaf_idx < static_cast<int32_t>(bsp.leafs.size())) {
          stats.visited_leafs++;
          const auto& leaf = bsp.leafs[static_cast<size_t>(leaf_idx)];
          const Vec3 mins{static_cast<float>(leaf.mins[0]), static_cast<float>(leaf.mins[1]), static_cast<float>(leaf.mins[2])};
          const Vec3 maxs{static_cast<float>(leaf.maxs[0]), static_cast<float>(leaf.maxs[1]), static_cast<float>(leaf.maxs[2])};
          if (AabbIntersectsFrustum(fr, mins, maxs)) {
            stats.accepted_leafs++;
          }
        }
      }
    }
  }
  return stats;
}

}  // namespace

TraversalStats RunCpuOptimizedTraversal(const BspData& bsp, const std::vector<Camera>& cameras, const int threads) {
  if (bsp.models.empty()) {
    throw std::runtime_error("BSP has no model data");
  }
  const NodeSoA soa = BuildNodeSoA(bsp);
  TraversalStats total{};

#ifdef _OPENMP
  if (threads > 0) {
    omp_set_num_threads(threads);
  }
#pragma omp parallel
  {
    TraversalStats local{};
#pragma omp for schedule(dynamic, 8)
    for (size_t i = 0; i < cameras.size(); ++i) {
      const auto s = TraverseCamera(bsp, soa, cameras[i]);
      local.visited_nodes += s.visited_nodes;
      local.visited_leafs += s.visited_leafs;
      local.culled_nodes += s.culled_nodes;
      local.accepted_leafs += s.accepted_leafs;
    }
#pragma omp critical
    {
      total.visited_nodes += local.visited_nodes;
      total.visited_leafs += local.visited_leafs;
      total.culled_nodes += local.culled_nodes;
      total.accepted_leafs += local.accepted_leafs;
    }
  }
#else
  (void)threads;
  for (const auto& cam : cameras) {
    const auto s = TraverseCamera(bsp, soa, cam);
    total.visited_nodes += s.visited_nodes;
    total.visited_leafs += s.visited_leafs;
    total.culled_nodes += s.culled_nodes;
    total.accepted_leafs += s.accepted_leafs;
  }
#endif
  return total;
}

}  // namespace quakecore

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

struct alignas(32) NodeSoA {
  std::vector<float> minx, miny, minz;
  std::vector<float> maxx, maxy, maxz;
  std::vector<int32_t> child0, child1;
  // Leaf AABBs pre-converted to float (avoids int16→float cast on the hot path).
  std::vector<float> leaf_minx, leaf_miny, leaf_minz;
  std::vector<float> leaf_maxx, leaf_maxy, leaf_maxz;
};

// All 6 frustum planes packed into 8-wide AVX2 layout; slots 6-7 zero-padded.
// Precomputed blend masks eliminate per-AABB sign comparisons.
struct alignas(32) FrustumSoA {
  // a, b, c components of the plane normals
  // for plane a*x + b*y + c*z + d = 0
  float nx[8];
  float ny[8];
  float nz[8];
  // negative distance from the origin to the plane
  float neg_d[8];  
  // chooses which corner of the AABB to use for the plane test
  float mask_x[8]; // 0x80000000 where nx<0 (pick maxx), else 0 (pick minx)
  float mask_y[8];
  float mask_z[8];
};

NodeSoA BuildNodeSoA(const BspData& bsp) {
  NodeSoA soa{};
  const size_t n = bsp.nodes.size();
  // Set the SoA vectors to match the number of nodes in the BSP
  soa.minx.resize(n);
  soa.miny.resize(n);
  soa.minz.resize(n);
  soa.maxx.resize(n);
  soa.maxy.resize(n);
  soa.maxz.resize(n);
  soa.child0.resize(n);
  soa.child1.resize(n);
  // Copy the node data into the SoA vectors.
  // node.mins/maxs are already float; node.children are int32.
  for (size_t i = 0; i < n; ++i) {
    const auto& node = bsp.nodes[i];
    soa.minx[i]   = node.mins[0];
    soa.miny[i]   = node.mins[1];
    soa.minz[i]   = node.mins[2];
    soa.maxx[i]   = node.maxs[0];
    soa.maxy[i]   = node.maxs[1];
    soa.maxz[i]   = node.maxs[2];
    soa.child0[i] = node.children[0];
    soa.child1[i] = node.children[1];
  }
  const size_t nl = bsp.leafs.size();
  soa.leaf_minx.resize(nl); 
  soa.leaf_miny.resize(nl); 
  soa.leaf_minz.resize(nl);
  soa.leaf_maxx.resize(nl); 
  soa.leaf_maxy.resize(nl); 
  soa.leaf_maxz.resize(nl);
  for (size_t i = 0; i < nl; ++i) {
    const auto& lf = bsp.leafs[i];
    soa.leaf_minx[i] = lf.mins[0];
    soa.leaf_miny[i] = lf.mins[1];
    soa.leaf_minz[i] = lf.mins[2];
    soa.leaf_maxx[i] = lf.maxs[0];
    soa.leaf_maxy[i] = lf.maxs[1];
    soa.leaf_maxz[i] = lf.maxs[2];
  }
  return soa;
}

inline void BuildFrustumSoA(const Frustum& fr, FrustumSoA& s) {
  static constexpr uint32_t kNegBit = 0x80000000u;
  for (int i = 0; i < 6; ++i) {
    const auto& p = fr.planes[i];
    s.nx[i]    = p.normal.x;
    s.ny[i]    = p.normal.y;
    s.nz[i]    = p.normal.z;
    s.neg_d[i] = -p.dist;
    uint32_t mx = (p.normal.x < 0.0f) ? kNegBit : 0u;
    uint32_t my = (p.normal.y < 0.0f) ? kNegBit : 0u;
    uint32_t mz = (p.normal.z < 0.0f) ? kNegBit : 0u;
    __builtin_memcpy(&s.mask_x[i], &mx, 4);
    __builtin_memcpy(&s.mask_y[i], &my, 4);
    __builtin_memcpy(&s.mask_z[i], &mz, 4);
  }
  // Slots 6 & 7: zero normal + zero neg_d → dot = 0, never culls.
  s.nx[6] = s.nx[7] = s.ny[6] = s.ny[7] = s.nz[6] = s.nz[7] = 0.0f;
  s.neg_d[6] = s.neg_d[7] = 0.0f;
  s.mask_x[6] = s.mask_x[7] = s.mask_y[6] = s.mask_y[7] = s.mask_z[6] = s.mask_z[7] = 0.0f;
}

// Test AABB against all 6 frustum planes in one AVX2+FMA pass.
// Returns true if AABB is visible (not culled by any plane).
inline bool AabbInFrustumAvx2(const FrustumSoA& fr,
                               const float minx, const float miny, const float minz,
                               const float maxx, const float maxy, const float maxz) {
  // Load the plane normals and distances into AVX2 registers
  const __m256 nx    = _mm256_load_ps(fr.nx);
  const __m256 ny    = _mm256_load_ps(fr.ny);
  const __m256 nz    = _mm256_load_ps(fr.nz);
  const __m256 neg_d = _mm256_load_ps(fr.neg_d);
  const __m256 mx    = _mm256_load_ps(fr.mask_x);
  const __m256 my_   = _mm256_load_ps(fr.mask_y);
  const __m256 mz    = _mm256_load_ps(fr.mask_z);

  // Chooses the correct corner of the AABB to use for the plane test
  // blendv: mask high-bit=1 → second arg (max); high-bit=0 → first arg (min).
  const __m256 rx = _mm256_blendv_ps(_mm256_set1_ps(minx), _mm256_set1_ps(maxx), mx);
  const __m256 ry = _mm256_blendv_ps(_mm256_set1_ps(miny), _mm256_set1_ps(maxy), my_);
  const __m256 rz = _mm256_blendv_ps(_mm256_set1_ps(minz), _mm256_set1_ps(maxz), mz);

  // dot[i] = nx[i]*rx[i] + ny[i]*ry[i] + nz[i]*rz[i] - dist[i]
  const __m256 dot = _mm256_fmadd_ps(nx, rx,
                       _mm256_fmadd_ps(ny, ry,
                         _mm256_fmadd_ps(nz, rz, neg_d)));

  // Culled if any lane has dot > 0 (reject corner is outside that plane).
  // __mm256_cmp_ps: performs dot[i] > 0 ? true : false for each entry i
  // _mm256_movemask_ps: extracts sign bit of each lane into 8-bit mask
  return _mm256_movemask_ps(_mm256_cmp_ps(dot, _mm256_setzero_ps(), _CMP_GT_OQ)) == 0;
}

// Scalar frustum-AABB test with early exit; faster than AVX2 when only 1-2 tests needed.
// Reimplementation of AabbIntersectsFrustum (frustum.cpp) to optimize for speed
inline bool AabbInFrustumScalar(const Frustum& fr,
                                const float minx, const float miny, const float minz,
                                const float maxx, const float maxy, const float maxz) {
  for (const auto& p : fr.planes) {
    const float rx = (p.normal.x >= 0.0f) ? minx : maxx;
    const float ry = (p.normal.y >= 0.0f) ? miny : maxy;
    const float rz = (p.normal.z >= 0.0f) ? minz : maxz;
    if (p.normal.x * rx + p.normal.y * ry + p.normal.z * rz - p.dist > 0.0f)
      return false;
  }
  return true;
}

TraversalStats TraverseCamera(const BspData& bsp, const NodeSoA& soa, const Camera& cam,
                              const Frustum* override_frustum) {
  TraversalStats stats{};
  const Frustum fr_raw = override_frustum ? *override_frustum : BuildFrustum(cam);

  const int32_t node_count = static_cast<int32_t>(soa.child0.size());
  const int32_t leaf_count = static_cast<int32_t>(soa.leaf_minx.size());
  const int32_t root = bsp.models.front().headnode[0];

  // If the root is not a valid node, return immediately
  if (root < 0 || root >= node_count) return stats;

  // Fast scalar pre-check on root: if root is culled, skip FrustumSoA entirely.
  // This matches baseline performance for the common case where cameras see nothing.
  stats.visited_nodes++;
  if (!AabbInFrustumScalar(fr_raw, soa.minx[root], soa.miny[root], soa.minz[root],
                                   soa.maxx[root], soa.maxy[root], soa.maxz[root])) {
    stats.culled_nodes++;
    return stats;
  }

  // If root visible: build FrustumSoA for the full AVX2 traversal.
  alignas(32) FrustumSoA fr;
  BuildFrustumSoA(fr_raw, fr);

  // Thread-local stack: capacity is retained across cameras, avoiding per-camera malloc. 
  // Stack is used to store the nodes to be visited.
  static thread_local std::vector<int32_t> tls_stack;
  tls_stack.clear();
  if (tls_stack.capacity() < 128) tls_stack.reserve(128);

  const int32_t c0r = soa.child0[root];
  const int32_t c1r = soa.child1[root];
  // Nodes are represented as positive indices for children, negative indices for leaves
  // Push the root's children onto the stack
  if (c0r >= 0) tls_stack.push_back(c0r);
  // If child0 is a leaf, check if it is visible
  else {
    // Leaves are encoded as -(leaf_index + 1)
    const int32_t li = -1 - c0r;
    if (li >= 0 && li < leaf_count) {
      stats.visited_leafs++;
      if (AabbInFrustumAvx2(fr, soa.leaf_minx[li], soa.leaf_miny[li], soa.leaf_minz[li],
                                 soa.leaf_maxx[li], soa.leaf_maxy[li], soa.leaf_maxz[li]))
        stats.accepted_leafs++;
    }
  }
  if (c1r >= 0) tls_stack.push_back(c1r);
  else {
    const int32_t li = -1 - c1r;
    if (li >= 0 && li < leaf_count) {
      stats.visited_leafs++;
      if (AabbInFrustumAvx2(fr, soa.leaf_minx[li], soa.leaf_miny[li], soa.leaf_minz[li],
                                 soa.leaf_maxx[li], soa.leaf_maxy[li], soa.leaf_maxz[li]))
        stats.accepted_leafs++;
    }
  }

  while (!tls_stack.empty()) {
    const int32_t idx = tls_stack.back();
    tls_stack.pop_back();
    if (idx < 0 || idx >= node_count) continue;

    stats.visited_nodes++;
    if (!AabbInFrustumAvx2(fr, soa.minx[idx], soa.miny[idx], soa.minz[idx],
                               soa.maxx[idx], soa.maxy[idx], soa.maxz[idx])) {
      stats.culled_nodes++;
      continue;
    }

    const int32_t c0 = soa.child0[idx];
    const int32_t c1 = soa.child1[idx];

    if (c0 >= 0) tls_stack.push_back(c0);
    else {
      const int32_t li = -1 - c0;
      if (li >= 0 && li < leaf_count) {
        stats.visited_leafs++;
        if (AabbInFrustumAvx2(fr, soa.leaf_minx[li], soa.leaf_miny[li], soa.leaf_minz[li],
                                   soa.leaf_maxx[li], soa.leaf_maxy[li], soa.leaf_maxz[li]))
          stats.accepted_leafs++;
      }
    }
    if (c1 >= 0) tls_stack.push_back(c1);
    else {
      const int32_t li = -1 - c1;
      if (li >= 0 && li < leaf_count) {
        stats.visited_leafs++;
        if (AabbInFrustumAvx2(fr, soa.leaf_minx[li], soa.leaf_miny[li], soa.leaf_minz[li],
                                   soa.leaf_maxx[li], soa.leaf_maxy[li], soa.leaf_maxz[li]))
          stats.accepted_leafs++;
      }
    }
  }
  return stats;
}

}  // namespace

TraversalStats RunCpuOptimizedTraversal(const BspData& bsp, const std::vector<Camera>& cameras, const int threads,
                                        const Frustum* override_frustum) {
  if (bsp.models.empty()) {
    throw std::runtime_error("BSP has no model data");
  }

  TraversalStats total{};

#ifdef _OPENMP
  if (threads > 0) {
    omp_set_num_threads(threads);
  }

  // Build the SoA inside the parallel region so threads warm up during construction
  // rather than during the traversal measurement itself.
  NodeSoA soa;
  uint64_t visited_nodes = 0, visited_leafs = 0, culled_nodes = 0, accepted_leafs = 0;

#pragma omp parallel reduction(+ : visited_nodes, visited_leafs, culled_nodes, accepted_leafs)
    {
#pragma omp single
      { soa = BuildNodeSoA(bsp); }
      // Implicit barrier: all threads are now warm and soa is ready.

#pragma omp for schedule(dynamic, 8) nowait
      for (size_t i = 0; i < cameras.size(); ++i) {
        const auto s = TraverseCamera(bsp, soa, cameras[i], override_frustum);
        visited_nodes  += s.visited_nodes;
        visited_leafs  += s.visited_leafs;
        culled_nodes   += s.culled_nodes;
        accepted_leafs += s.accepted_leafs;
      }
    }
    total.visited_nodes  = visited_nodes;
    total.visited_leafs  = visited_leafs;
    total.culled_nodes   = culled_nodes;
    total.accepted_leafs = accepted_leafs;
#else
  (void)threads;
  const NodeSoA soa = BuildNodeSoA(bsp);
  for (const auto& cam : cameras) {
    const auto s = TraverseCamera(bsp, soa, cam, override_frustum);
    total.visited_nodes  += s.visited_nodes;
    total.visited_leafs  += s.visited_leafs;
    total.culled_nodes   += s.culled_nodes;
    total.accepted_leafs += s.accepted_leafs;
  }
#endif
  return total;
}

}  // namespace quakecore

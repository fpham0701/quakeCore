#include "quakecore/engines.hpp"

#include <stdexcept>

#include "quakecore/frustum.hpp"

namespace quakecore {
namespace {

void TraverseRecursive(const BspData& bsp, const Frustum& frustum, int node_index, TraversalStats& stats) {
  if (node_index < 0 || node_index >= static_cast<int>(bsp.nodes.size())) {
    return;
  }
  stats.visited_nodes++;
  const auto& node = bsp.nodes[static_cast<size_t>(node_index)];
  const Vec3 mins{node.mins[0], node.mins[1], node.mins[2]};
  const Vec3 maxs{node.maxs[0], node.maxs[1], node.maxs[2]};
  if (!AabbIntersectsFrustum(frustum, mins, maxs)) {
    stats.culled_nodes++;
    return;
  }

  for (int c = 0; c < 2; ++c) {
    const int child = node.children[c];
    if (child >= 0) {
      TraverseRecursive(bsp, frustum, child, stats);
    } else {
      const int leaf_index = -1 - child;
      if (leaf_index >= 0 && leaf_index < static_cast<int>(bsp.leafs.size())) {
        stats.visited_leafs++;
        const auto& leaf = bsp.leafs[static_cast<size_t>(leaf_index)];
        const Vec3 lmins{leaf.mins[0], leaf.mins[1], leaf.mins[2]};
        const Vec3 lmaxs{leaf.maxs[0], leaf.maxs[1], leaf.maxs[2]};
        if (AabbIntersectsFrustum(frustum, lmins, lmaxs)) {
          stats.accepted_leafs++;
        }
      }
    }
  }
}

}  // namespace

TraversalStats RunBaselineTraversal(const BspData& bsp, const std::vector<Camera>& cameras) {
  if (bsp.models.empty()) {
    throw std::runtime_error("BSP has no model data");
  }
  TraversalStats total{};
  const int root = bsp.models.front().headnode[0];
  for (const auto& cam : cameras) {
    const Frustum fr = BuildFrustum(cam);
    TraverseRecursive(bsp, fr, root, total);
  }
  return total;
}

}  // namespace quakecore

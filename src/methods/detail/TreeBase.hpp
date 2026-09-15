#pragma once
#include <queue>
#include <vector>

#include "Bitset.hpp"
#include "Graph.hpp"

namespace algconn::detail {

// Deterministic BFS spanning tree of a connected graph, as a pair-index
// bitset of exactly n-1 edges. Root 0, neighbors visited in ascending order.
//
// The completion-style baselines ported from the CDC 2026 companion code
// (MDMD, FVG-build, k-opt exchange) construct or search over graphs that
// contain a fixed connected "base". Under this paper's protocol every method
// starts from the same G0 in G_{n,m}; taking the base to be a spanning tree
// OF G0 honors that protocol and makes connectivity automatic -- the tree is
// never touched, so every graph these methods ever hold is connected.
inline Bitset bfsSpanningTree(const Graph& g) {
  const PairIndex& pix = g.pix();
  const int n = g.n();
  std::vector<std::vector<int>> adj(n);
  for (auto [u, v] : g.edgeList()) { adj[u].push_back(v); adj[v].push_back(u); }

  Bitset tree(pix.numPairs());
  std::vector<char> seen(n, 0);
  std::queue<int> q;
  seen[0] = 1;
  q.push(0);
  while (!q.empty()) {
    const int u = q.front();
    q.pop();
    for (int v : adj[u]) {
      if (seen[v]) continue;
      seen[v] = 1;
      tree.set(pix.index(u, v));
      q.push(v);
    }
  }
  return tree;
}

}  // namespace algconn::detail

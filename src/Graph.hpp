#pragma once
#include <vector>

#include "Bitset.hpp"
#include "PairIndex.hpp"
#include "Rng.hpp"
#include "Swap.hpp"

namespace algconn {

// A member of G_{n,m}: exactly m edges, simple, connected.
//
// The class exposes NO operation that changes the edge count except applySwap,
// which adds one pair and removes another in the same call. Leaving G_{n,m} by
// accident is therefore not expressible, which is what makes every method
// built on this class a rewiring method.
class Graph {
 public:
  Graph(const PairIndex& pix, Bitset edges);

  // Tree-seeded: random spanning tree (n-1 edges) plus exactly m-(n-1) extra
  // pairs. Exact count by construction; connected because it contains a tree.
  static Graph randomConnected(const PairIndex& pix, int m, Rng& rng);

  // Reference initialisation (build_ring_random in the upstream C code):
  // shuffle the vertices, join them as a PATH (n-1 edges), then add a uniformly
  // random (m-n+1)-subset of the remaining pairs.
  static Graph randomPathSeeded(const PairIndex& pix, int m, Rng& rng);

  int n() const { return pix_->n(); }
  int m() const { return m_; }
  int numPairs() const { return pix_->numPairs(); }
  const PairIndex& pix() const { return *pix_; }
  const Bitset& edges() const { return e_; }

  bool has(int pair) const { return e_.test(pair); }
  int degree(int v) const { return deg_[v]; }
  const std::vector<int>& degrees() const { return deg_; }

  // The only mutator. Asserts legality (add absent, remove present).
  void applySwap(Swap s);

  bool isConnected() const;
  // Pair indices whose removal disconnects the graph (Tarjan, O(n+m)).
  std::vector<int> bridgePairs() const;
  // Whether applying s leaves the graph connected.
  bool swapKeepsConnected(Swap s) const;

  // Dense Laplacian, row-major n*n.
  std::vector<double> laplacian() const;

  // --- zero-oracle-cost spectral information -----------------------------
  // tr(L) = 2m is constant on G_{n,m}; tr(L^2) = sum_i d_i^2 + 2m depends only
  // on the degree sequence. Both are maintained incrementally by applySwap.
  long sumDegSquared() const { return sumDegSq_; }
  // lambda_2 <= 2m/(n-1) - sigma/sqrt(n-2), certified, no eigensolve.
  double momentUpperBound() const;

  std::vector<std::pair<int, int>> edgeList() const;

  // Throws std::logic_error if any invariant of G_{n,m} is violated.
  void checkFeasible() const;

 private:
  void rebuildDegrees();

  const PairIndex* pix_;
  Bitset e_;
  int m_ = 0;
  std::vector<int> deg_;
  long sumDegSq_ = 0;
};

// Exact completed-swap connectivity oracle for one fixed connected graph.
//
// A swap can disconnect G only when its removed edge is a bridge. Root one DFS
// tree. If bridge e has child c, G-e has the DFS subtree of c on one side and
// every other vertex on the other; adding (u,v) reconnects the swap exactly
// when u and v lie on opposite sides. DFS intervals therefore give O(1) exact
// queries after O(n+m+binom(n,2)) preprocessing. No numerical resistance
// threshold is involved.
class SwapConnectivity {
 public:
  explicit SwapConnectivity(const Graph& graph);
  bool keepsConnected(Swap swap) const;

 private:
  const PairIndex* pix_ = nullptr;
  std::vector<int> entry_, exit_, bridgeChild_;
};

}  // namespace algconn

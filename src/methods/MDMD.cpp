#include "methods/MDMD.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <vector>

#include "methods/detail/TreeBase.hpp"

namespace algconn {
namespace {

// Adjacency lists from a pair bitset; neighbors ascending (determinism).
std::vector<std::vector<int>> adjacency(const PairIndex& pix, const Bitset& e) {
  std::vector<std::vector<int>> adj(pix.n());
  for (int k = 0; k < pix.numPairs(); ++k)
    if (e.test(k)) { adj[pix.i(k)].push_back(pix.j(k)); adj[pix.j(k)].push_back(pix.i(k)); }
  return adj;
}

std::vector<int> bfsDistances(const std::vector<std::vector<int>>& adj, int src) {
  std::vector<int> dist(adj.size(), -1);
  std::queue<int> q;
  dist[src] = 0;
  q.push(src);
  while (!q.empty()) {
    const int u = q.front();
    q.pop();
    for (int v : adj[u])
      if (dist[v] < 0) { dist[v] = dist[u] + 1; q.push(v); }
  }
  return dist;
}

// Reference tie-breaking (`_choose_with_ec`, random_tie = false): among the
// candidates, keep those of minimum EC (if given), then take the least index.
int chooseWithEc(const std::vector<int>& cand, const std::vector<long>* ec) {
  int best = cand.front();
  if (ec == nullptr) {
    for (int c : cand)
      if (c < best) best = c;
    return best;
  }
  long bestEc = std::numeric_limits<long>::max();
  for (int c : cand) {
    const long e = (*ec)[c];
    if (e < bestEc || (e == bestEc && c < best)) { bestEc = e; best = c; }
  }
  return best;
}

}  // namespace

Result MDMD::run(Graph start, Oracle& oracle, Rng& rng) const {
  (void)rng;  // deterministic
  const auto t0 = std::chrono::steady_clock::now();
  const PairIndex& pix = start.pix();
  const int n = start.n();
  const int m = start.m();

  Bitset e = detail::bfsSpanningTree(start);
  int edges = n - 1;

  std::vector<int> deg(n, 1);  // tree degrees, rebuilt below for exactness
  {
    std::fill(deg.begin(), deg.end(), 0);
    for (int k = 0; k < pix.numPairs(); ++k)
      if (e.test(k)) { ++deg[pix.i(k)]; ++deg[pix.j(k)]; }
  }

  int steps = 0;
  while (edges < m) {
    const auto adj = adjacency(pix, e);

    // EC = sum of neighbor degrees (reference: A @ deg).
    std::vector<long> ec(n, 0);
    if (cfg_.useEc)
      for (int u = 0; u < n; ++u)
        for (int v : adj[u]) ec[u] += deg[v];

    int minDeg = std::numeric_limits<int>::max();
    for (int u = 0; u < n; ++u) minDeg = std::min(minDeg, deg[u]);
    std::vector<int> minSet;
    for (int u = 0; u < n; ++u)
      if (deg[u] == minDeg) minSet.push_back(u);
    const int i0 = chooseWithEc(minSet, cfg_.useEc ? &ec : nullptr);

    const std::vector<int> dist = bfsDistances(adj, i0);
    int dMax = -1;
    std::vector<int> far;
    for (int v = 0; v < n; ++v) {
      if (v == i0 || e.test(pix.index(i0, v))) continue;
      if (dist[v] > dMax) { dMax = dist[v]; far.clear(); }
      if (dist[v] == dMax) far.push_back(v);
    }
    int i1 = i0, j1 = -1;
    if (!far.empty()) {
      j1 = chooseWithEc(far, cfg_.useEc ? &ec : nullptr);
    } else {
      // Defensive fallback mirroring the reference: i0 has no non-neighbor
      // (impossible while edges < m unless the graph is complete); take the
      // lexicographically first absent pair instead.
      for (int k = 0; k < pix.numPairs() && j1 < 0; ++k)
        if (!e.test(k)) { i1 = pix.i(k); j1 = pix.j(k); }
      if (j1 < 0) break;
    }

    const int k = pix.index(i1, j1);
    e.set(k);
    ++deg[i1];
    ++deg[j1];
    ++edges;
    ++steps;
  }

  Graph out(pix, std::move(e));
  const double lam = oracle.lam2(out);  // the constructor's single oracle call
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out.checkFeasible();
  return Result{name(),        out.n(), out.m(), lam,
                out.edgeList(), oracle.calls(), oracle.queries(),
                wall,          0,       steps,   out.momentUpperBound(),
                oracle.multiplicity(out)};
}

}  // namespace algconn

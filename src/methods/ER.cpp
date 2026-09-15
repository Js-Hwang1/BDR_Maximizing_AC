#include "methods/ER.hpp"

#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>
#include <vector>

#include "methods/detail/GroundedInverse.hpp"

namespace algconn {
namespace {

// FNV over the edge bitset words: the same keying the oracle uses. A hash
// collision only ends a walk early, which costs exploration, never
// correctness -- the method returns only graphs it has evaluated.
uint64_t hashEdges(const Bitset& e) {
  uint64_t h = 1469598103934665603ULL;
  for (uint64_t w : e.words()) { h ^= w; h *= 1099511628211ULL; }
  return h;
}

}  // namespace

Result ER::run(Graph start, Oracle& oracle, Rng& rng) const {
  const auto t0 = std::chrono::steady_clock::now();
  const PairIndex& pix = start.pix();
  const int n = start.n();
  const int m = start.m();
  const int maxM = pix.numPairs();

  Graph cur = start;
  Graph bestG = start;
  double bestLam = oracle.lam2(start);  // anchor: the shared G0 is call one
  long moves = 0;
  int restarts = 0, stagnant = 0;

  auto freeBound = [&](const Graph& g) {
    int dmin = n;
    for (int v = 0; v < n; ++v) dmin = std::min(dmin, g.degree(v));
    const double fied = (m == maxM) ? (double)n : (double)dmin;
    return std::min(fied, g.momentUpperBound());
  };

  while (!oracle.exhausted() && restarts < cfg_.maxRestarts) {
    detail::GroundedInverse inv(pix, cur.edges(), 0, cfg_.exactResetEvery, cfg_.ridge);
    std::unordered_set<uint64_t> visited;
    visited.insert(hashEdges(cur.edges()));
    const long callsAtWalk = oracle.calls();

    std::vector<char> isBridge(pix.numPairs());
    for (long step = 0; step < cfg_.maxWalk; ++step) {
      // Top-scored swap on the CURRENT graph, deterministic tie-break by pair
      // index: add the absent pair of largest resistance, remove the
      // non-bridge present edge of smallest. Bridges come from Tarjan on the
      // exact structure, not from the drifting inverse: a Sherman-Morrison
      // error of 1e-6 near R_eff = 1 must not disconnect the walk.
      std::fill(isBridge.begin(), isBridge.end(), 0);
      for (int k : cur.bridgePairs()) isBridge[k] = 1;
      int bestAdd = -1, bestRem = -1;
      double rAdd = -1.0, rRem = 2.0;
      for (int k = 0; k < pix.numPairs(); ++k) {
        const double r = inv.effectiveResistance(pix.i(k), pix.j(k));
        if (cur.has(k)) {
          if (r < rRem && !isBridge[k]) { rRem = r; bestRem = k; }
        } else if (r > rAdd) {
          rAdd = r; bestAdd = k;
        }
      }
      if (bestAdd < 0 || bestRem < 0) break;  // no legal swap: end the walk

      Swap s{bestAdd, bestRem};
      Graph trial = cur;
      trial.applySwap(s);
      Bitset mid = cur.edges();
      mid.clear(s.remove);
      inv.removeEdge(pix.i(s.remove), pix.j(s.remove), mid);
      inv.addEdge(pix.i(s.add), pix.j(s.add), trial.edges());
      cur = std::move(trial);
      ++moves;

      if (!visited.insert(hashEdges(cur.edges())).second) break;  // cycle

      if (freeBound(cur) > bestLam - cfg_.tol && !oracle.exhausted()) {
        const double lam = oracle.lam2(cur);
        if (lam > bestLam) { bestLam = lam; bestG = cur; }
      }
    }

    ++restarts;
    if (oracle.calls() == callsAtWalk) {
      if (++stagnant >= cfg_.stagnationLimit) break;  // walks buy nothing new
    } else {
      stagnant = 0;
    }
    if (oracle.exhausted()) break;
    cur = Graph::randomConnected(pix, m, rng);
  }

  bestG.checkFeasible();
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return Result{name(),           bestG.n(),      bestG.m(),        bestLam,
                bestG.edgeList(), oracle.calls(), oracle.queries(),
                wall,             0,              (int)std::min<long>(moves, 1 << 30),
                bestG.momentUpperBound(), oracle.multiplicity(bestG)};
}

}  // namespace algconn

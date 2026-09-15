#include "methods/KOPT.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#include "methods/detail/TreeBase.hpp"

namespace algconn {
namespace {

// All size-k combinations of `pool` (by index into it), capped: if the exact
// count exceeds `cap`, sample `cap` distinct combinations with the generator
// (mirrors the reference's `_choose_edge_combos`). For k = 1 this is simply
// the singletons, deterministic.
std::vector<std::vector<int>> combos(const std::vector<int>& pool, int k, int cap,
                                     Rng& rng) {
  std::vector<std::vector<int>> out;
  const int p = static_cast<int>(pool.size());
  if (k <= 0 || p < k) return out;
  if (k == 1) {
    for (int i = 0; i < p && static_cast<int>(out.size()) < cap; ++i)
      out.push_back({pool[i]});
    return out;
  }
  // Exact enumeration while below cap; otherwise sampled k-subsets.
  double total = 1.0;
  for (int i = 0; i < k; ++i) total *= static_cast<double>(p - i) / (i + 1);
  if (total <= cap) {
    std::vector<int> idx(k);
    for (int i = 0; i < k; ++i) idx[i] = i;
    while (true) {
      std::vector<int> c(k);
      for (int i = 0; i < k; ++i) c[i] = pool[idx[i]];
      out.push_back(std::move(c));
      int i = k - 1;
      while (i >= 0 && idx[i] == p - k + i) --i;
      if (i < 0) break;
      ++idx[i];
      for (int t = i + 1; t < k; ++t) idx[t] = idx[t - 1] + 1;
    }
    return out;
  }
  std::vector<int> scratch(pool);
  for (int s = 0; s < cap; ++s) {
    for (int i = 0; i < k; ++i) {
      const int r = i + static_cast<int>(rng.below(static_cast<uint32_t>(p - i)));
      std::swap(scratch[i], scratch[r]);
    }
    std::vector<int> c(scratch.begin(), scratch.begin() + k);
    std::sort(c.begin(), c.end());
    out.push_back(std::move(c));
  }
  return out;
}

// Pairs ranked by Fiedler gap of v2; descending for additions, ascending for
// removals. Ties by pair index (stable, reproducible).
std::vector<int> rankByGap(const PairIndex& pix, const std::vector<double>& v2,
                           const std::vector<int>& pairs, bool descending) {
  std::vector<std::pair<double, int>> scored;
  scored.reserve(pairs.size());
  for (int k : pairs) {
    const double d = v2[pix.i(k)] - v2[pix.j(k)];
    scored.push_back({d * d, k});
  }
  std::sort(scored.begin(), scored.end(),
            [descending](const auto& a, const auto& b) {
              if (a.first != b.first)
                return descending ? a.first > b.first : a.first < b.first;
              return a.second < b.second;
            });
  std::vector<int> out;
  out.reserve(scored.size());
  for (auto& [s, k] : scored) out.push_back(k);
  return out;
}

}  // namespace

Result KOPT::run(Graph start, Oracle& oracle, Rng& rng) const {
  const auto t0 = std::chrono::steady_clock::now();
  const PairIndex& pix = start.pix();
  const int np = pix.numPairs();
  const int m = start.m();

  const Bitset tree = detail::bfsSpanningTree(start);

  // Augment set: the start graph's non-tree pairs (so the search starts at G0).
  std::vector<int> selected;
  for (int k = 0; k < np; ++k)
    if (start.edges().test(k) && !tree.test(k)) selected.push_back(k);

  Graph cur = start;
  double curLam = oracle.lam2(cur);
  int rounds = 0;

  // Best over every restart, since a restart may land worse than where it left.
  Graph bestG = cur;
  double bestOverall = curLam;
  int restarts = 0, stagnant = 0;
  long callsAtRestart = oracle.calls();

  for (int round = 0; round < cfg_.maxRounds; ++round) {
    if (oracle.exhausted()) break;

    // Add-side ranking on the current graph: free screening from its call.
    const std::vector<double> v2 = oracle.eigenpair(cur).v2;
    std::vector<int> addAvail;
    for (int k = 0; k < np; ++k)
      if (!cur.edges().test(k)) addAvail.push_back(k);
    if (static_cast<int>(addAvail.size()) < cfg_.k) break;

    std::vector<int> addRanked = rankByGap(pix, v2, addAvail, /*descending=*/true);
    if (static_cast<int>(addRanked.size()) > cfg_.pool) addRanked.resize(cfg_.pool);
    const auto addCombos = combos(addRanked, cfg_.k, cfg_.comboCapAdd, rng);
    if (addCombos.empty()) break;

    double bestLam = curLam;
    std::vector<int> bestSelected;
    Bitset bestEdges;

    for (const auto& addC : addCombos) {
      if (oracle.exhausted()) break;
      // Augmented graph: current plus the added pairs (m + k edges).
      Bitset plus = cur.edges();
      for (int k : addC) plus.set(k);
      Graph plusG(pix, plus);
      const std::vector<double> vplus =
          oracle.eigenpair(plusG).v2;  // counted off-manifold guidance

      std::vector<int> selPlus(selected);
      selPlus.insert(selPlus.end(), addC.begin(), addC.end());
      std::sort(selPlus.begin(), selPlus.end());
      std::vector<int> remRanked =
          rankByGap(pix, vplus, selPlus, /*descending=*/false);
      if (static_cast<int>(remRanked.size()) > cfg_.pool) remRanked.resize(cfg_.pool);
      const auto remCombos = combos(remRanked, cfg_.k, cfg_.comboCapDel, rng);

      for (const auto& remC : remCombos) {
        if (oracle.exhausted()) break;
        // Skip degenerate exchanges that delete what was just added.
        bool overlaps = false;
        for (int r : remC)
          for (int a : addC)
            if (r == a) overlaps = true;
        if (overlaps) continue;

        Bitset cand = plus;
        for (int r : remC) cand.clear(r);
        Graph candG(pix, cand);  // back in G_{n,m}; connected (tree intact)
        const double lam = oracle.lam2(candG);
        if (lam > bestLam + 1e-12) {
          bestLam = lam;
          bestEdges = cand;
          bestSelected.clear();
          for (int k : selPlus) {
            bool removed = false;
            for (int r : remC) removed |= (r == k);
            if (!removed) bestSelected.push_back(k);
          }
        }
      }
    }

    if (bestSelected.empty()) {
      // Converged: no improving k-exchange exists in this neighborhood, which
      // is where the paper stops. Under a budget there is still budget left,
      // so restart from a fresh random graph and keep the best -- the same
      // rule FV and ER use.
      if (curLam > bestOverall) { bestOverall = curLam; bestG = cur; }
      // Restart only when a budget was actually set. With an unbounded oracle
      // there is nothing to spend restarts against, and since every restart
      // evaluates a fresh random graph it always buys a new call -- so the
      // stagnation counter never fires and the loop would run to maxRestarts.
      // Unbounded means "run the paper's local search", which stops right
      // here, at the first graph admitting no improving k-exchange.
      if (oracle.budget() < 0) break;
      if (oracle.calls() == callsAtRestart) {
        if (++stagnant >= cfg_.stagnationLimit) break;  // nothing new to learn
      } else {
        stagnant = 0;
      }
      if (++restarts >= cfg_.maxRestarts || oracle.exhausted()) break;
      callsAtRestart = oracle.calls();
      cur = Graph::randomConnected(pix, m, rng);
      const Bitset t2 = detail::bfsSpanningTree(cur);
      selected.clear();
      for (int q = 0; q < np; ++q)
        if (cur.edges().test(q) && !t2.test(q)) selected.push_back(q);
      curLam = oracle.lam2(cur);
      continue;
    }
    selected = std::move(bestSelected);
    cur = Graph(pix, bestEdges);
    curLam = bestLam;
    if (curLam > bestOverall) { bestOverall = curLam; bestG = cur; }
    ++rounds;
  }
  if (curLam > bestOverall) { bestOverall = curLam; bestG = cur; }

  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bestG.checkFeasible();
  return Result{name(),          bestG.n(),      bestG.m(), bestOverall,
                bestG.edgeList(), oracle.calls(), oracle.queries(),
                wall,            0,              rounds,    bestG.momentUpperBound(),
                oracle.multiplicity(bestG)};
}

}  // namespace algconn

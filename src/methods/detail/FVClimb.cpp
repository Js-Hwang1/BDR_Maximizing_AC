#include "methods/detail/FVClimb.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

namespace algconn {

Result FVClimb::run(Graph g, Oracle& oracle, Rng& rng) const {
  (void)rng;  // deterministic given the start
  const auto t0 = std::chrono::steady_clock::now();
  const PairIndex& pix = g.pix();
  const int np = pix.numPairs();

  // Hoisted out of the step loop: these were the dominant allocation cost.
  std::vector<char> isBridge(np, 0);
  std::vector<std::pair<double, int>> addC, remC;
  addC.reserve(np);
  remC.reserve(np);

  double best = oracle.lam2(g);
  int steps = 0;

  for (int it = 0; it < cfg_.maxSteps; ++it) {
    if (oracle.exhausted()) break;
    const std::vector<double> v2 = oracle.eigenpair(g).v2;

    std::fill(isBridge.begin(), isBridge.end(), 0);
    for (int k : g.bridgePairs()) isBridge[k] = 1;

    // Fiedler gap of every pair -- free, from the v2 the oracle already gave.
    addC.clear();
    remC.clear();
    for (int k = 0; k < np; ++k) {
      const double d = v2[pix.i(k)] - v2[pix.j(k)];
      const double gap = d * d;
      if (g.has(k)) {
        if (!isBridge[k]) remC.push_back({gap, k});  // never delete a bridge
      } else {
        addC.push_back({gap, k});
      }
    }
    if (addC.empty() || remC.empty()) break;

    // Only the head of each ordering is ever used, so partial_sort suffices.
    const int na = std::min<int>(cfg_.candidates, static_cast<int>(addC.size()));
    const int nr = std::min<int>(cfg_.candidates, static_cast<int>(remC.size()));
    // Ties in the Fiedler gap are common -- they are precisely the symptom of
    // a degenerate lambda_2 -- and the climb's outcome is genuinely sensitive
    // to how they are broken. Break them by pair index so the algorithm is
    // well defined and reproducible across implementations.
    std::partial_sort(addC.begin(), addC.begin() + na, addC.end(),
                      [](const auto& a, const auto& b) {
                        return a.first != b.first ? a.first > b.first
                                                  : a.second < b.second;
                      });
    std::partial_sort(remC.begin(), remC.begin() + nr, remC.end(),
                      [](const auto& a, const auto& b) {
                        return a.first != b.first ? a.first < b.first
                                                  : a.second < b.second;
                      });

    bool improved = false;
    for (int ia = 0; ia < na && !improved; ++ia) {
      for (int ir = 0; ir < nr && !improved; ++ir) {
        if (oracle.exhausted()) break;
        // No connectivity check is needed: remC excludes bridges, and removing
        // a non-bridge from a connected graph leaves it connected, so the swap
        // is legal for ANY inserted pair. (Guarded by a test.)
        Graph cand = g;
        cand.applySwap(Swap{addC[ia].second, remC[ir].second});
        const double lam = oracle.lam2(cand);
        if (lam > best + 1e-12) {
          g = std::move(cand);
          best = lam;
          improved = true;
          ++steps;
        }
      }
    }
    if (!improved) break;  // leaf: no first-order candidate improves lambda_2
  }

  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  g.checkFeasible();
  return Result{name(),        g.n(),  g.m(),  best,
                g.edgeList(),  oracle.calls(), oracle.queries(),
                wall,          0,      steps,  g.momentUpperBound(),
                oracle.multiplicity(g)};
}

}  // namespace algconn

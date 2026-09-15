#include "methods/RANDOM.hpp"

#include <chrono>
#include <utility>
#include <vector>

#include "methods/detail/TreeBase.hpp"

namespace algconn {

Result RANDOM::run(Graph start, Oracle& oracle, Rng& rng) const {
  const auto t0 = std::chrono::steady_clock::now();
  const PairIndex& pix = start.pix();
  const int m = start.m();

  Bitset e = detail::bfsSpanningTree(start);
  int edges = start.n() - 1;

  // The reference rebuilds the absent-pair list every step and indexes into it
  // uniformly. Building it once and swap-removing the chosen entry draws from
  // the same uniform distribution over absent pairs at every step, but turns
  // an O(n^2) rescan per edge into O(1).
  std::vector<int> absent;
  absent.reserve(pix.numPairs());
  for (int k = 0; k < pix.numPairs(); ++k)
    if (!e.test(k)) absent.push_back(k);

  int steps = 0;
  while (edges < m && !absent.empty()) {
    const uint32_t at = rng.below(static_cast<uint32_t>(absent.size()));
    const int k = absent[at];
    absent[at] = absent.back();
    absent.pop_back();
    e.set(k);
    ++edges;
    ++steps;
  }

  Graph out(pix, std::move(e));
  const double lam = oracle.lam2(out);
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out.checkFeasible();
  return Result{name(),         out.n(),        out.m(),  lam,
                out.edgeList(), oracle.calls(), oracle.queries(),
                wall,           0,              steps,    out.momentUpperBound(),
                oracle.multiplicity(out)};
}

}  // namespace algconn

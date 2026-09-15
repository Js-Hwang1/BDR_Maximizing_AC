#include "methods/FV.hpp"

#include <chrono>
#include <utility>

#include "methods/detail/FVClimb.hpp"

namespace algconn {
namespace {

Graph fromEdges(const PairIndex& pix, const std::vector<std::pair<int, int>>& es) {
  Bitset b(pix.numPairs());
  for (auto [u, v] : es) b.set(pix.index(u, v));
  return Graph(pix, std::move(b));
}

}  // namespace

Result FV::run(Graph start, Oracle& oracle, Rng& rng) const {
  const auto t0 = std::chrono::steady_clock::now();
  const PairIndex& pix = start.pix();
  const int m = start.m();

  Result best;
  best.lam2 = -1.0;
  int restarts = 0;
  int stagnant = 0;
  int width = cfg_.candidates;
  Graph current = start;  // first climb uses the shared per-seed start

  while (!oracle.exhausted() && restarts < cfg_.maxRestarts) {
    const long before = oracle.calls();
    Result r;
    try {
      FVClimb::Config gc;
      gc.candidates = width;
      r = FVClimb(gc).run(current, oracle, rng);
    } catch (const std::exception&) {
      break;  // budget ran out mid-climb
    }
    if (r.lam2 > best.lam2) best = r;
    if (oracle.exhausted()) break;

    // The climb has halted. Ask the certificate whether that halt means
    // anything: r.multiplicity is the multiplicity of lambda_2 AT THE LEAF,
    // already paid for by the call that ended the climb.
    const Graph leaf = fromEdges(pix, r.edges);
    if (r.multiplicity < 2 && width < cfg_.maxCandidates) {
      // lambda_2 simple: prop:degen certifies nothing, so the halt may be an
      // artifact of the candidate width. Widen and resume from the same graph.
      width = std::min(cfg_.maxCandidates, width * 2);
      current = leaf;
      continue;
    }

    ++restarts;
    width = cfg_.candidates;
    if (oracle.calls() == before) {
      if (++stagnant >= cfg_.stagnationLimit) break;  // search space exhausted
    } else {
      stagnant = 0;
    }
    if (oracle.exhausted()) break;
    current = Graph::randomConnected(pix, m, rng);
  }

  best.method = name();
  best.steps = restarts;
  best.oracleCalls = oracle.calls();
  best.oracleQueries = oracle.queries();
  best.wallSec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return best;
}

}  // namespace algconn

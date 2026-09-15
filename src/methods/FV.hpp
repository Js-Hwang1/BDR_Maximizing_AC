#pragma once
#include "Method.hpp"

namespace algconn {

// FV-greedy climbs to its first leaf and stops, spending only O(depth) oracle
// calls. Comparing that against a population method at its own, far larger
// budget would be meaningless. FV is the budget-matched baseline: climb,
// and on reaching a leaf either widen or restart, until the budget is spent.
//
// The restart is gated on the DEGENERACY CERTIFICATE. At a leaf the
// multiplicity of lambda_2 is already known -- it comes from a call that has
// been paid for. If lambda_2 is degenerate then by Proposition~\ref{prop:degen}
// no single swap can improve it: the leaf is a certified local optimum and the
// only useful move is to restart elsewhere. If lambda_2 is simple the
// proposition certifies nothing, and the halt may merely be an artifact of the
// finite candidate list, so the width is doubled and the climb resumed from the
// same graph before a restart is spent. This is the theorem doing work inside
// the algorithm rather than only in the write-up.
//
// This is what the paper's protocol requires -- same start, same neighborhood,
// SAME B -- so any remaining gap is attributable to search strategy.
class FV : public Method {
 public:
  struct Config {
    // A restart that buys no NEW oracle call has told us nothing: every graph
    // it touched was already cached. After this many such restarts in a row,
    // stop. Budget exhaustion alone is NOT a sufficient termination condition,
    // because memoized evaluations never consume budget -- and near
    // m = binom(n,2) the feasible set is small enough to exhaust outright
    // (at m = binom(n,2)-1 there are only binom(n,2) graphs in total).
    int stagnationLimit = 25;
    int maxRestarts = 100000;
    int candidates = 8;        // initial FVClimb candidate width
    int maxCandidates = 128;   // cap before a non-degenerate leaf restarts
  };

  FV() = default;
  explicit FV(Config cfg) : cfg_(cfg) {}
  std::string name() const override { return "fv"; }
  Result run(Graph start, Oracle& oracle, Rng& rng) const override;

 private:
  Config cfg_{};
};

}  // namespace algconn

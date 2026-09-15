#pragma once
#include "Method.hpp"

namespace algconn {

// ER: the effective-resistance rule as a FREE score-greedy walk. This is THE
// ER of this work, and the only one: there is no verification variant and no
// flag selecting between them. An earlier ER paid one eigensolve per proposed
// swap and accepted only on a measured increase in lambda_2; that is a
// lambda_2 hill-climber that happens to order its proposals by resistance,
// not the resistance heuristic. It was replaced, not parameterised, and
// survives only in git history (through commit 8b5fa66). Tapes it produced
// carried the method name "er_rewire" and have been purged from results/.
//
// The grounded Laplacian inverse is maintained by Sherman-Morrison from the
// graph structure alone, so the ER score consults no oracle: under the model
// of Section II, every move is free computation. The walk therefore accepts
// the top-scored swap unconditionally -- add the absent pair of largest
// effective resistance, remove the non-bridge edge of smallest -- with no
// per-move verification. Of the two classical screening rules only this one
// survives removal of the oracle from the loop; the Fiedler rule needs v_2 of
// the current graph, which only a paid call provides.
//
// TERMINATION, free: the walk is deterministic (argmax with tie-break by pair
// index), the space finite, so every walk enters a cycle; the first revisit
// of a stored graph hash ends the walk and triggers a restart from a fresh
// random graph. No multiplicity check, no width ladder.
//
// PAYMENT, gated: unverified moves do not know lambda_2, and a method must
// return a graph it has evaluated. A visited graph is evaluated only when its
// free certificate allows it to beat the incumbent:
//     min( delta_min  [Fiedler, non-complete],  moment bound )  >  best - tol.
// This is the same prune that carried the exact n=12 enumeration. The budget
// is thus spent only on plausibly-optimal visits, and B buys on the order of
// B basins instead of B/(depth + halt) climbs.
class ER : public Method {
 public:
  struct Config {
    int exactResetEvery = 32;   // grounded-inverse refactorization period
    double ridge = 1e-10;
    double tol = 1e-9;          // gate tolerance against the incumbent
    long maxWalk = 200000;      // safety cap per walk; treated as a cycle
    int stagnationLimit = 50;   // consecutive walks buying no evaluation
    int maxRestarts = 10000000;
  };

  ER() = default;
  explicit ER(Config cfg) : cfg_(cfg) {}
  std::string name() const override { return "er"; }
  Result run(Graph start, Oracle& oracle, Rng& rng) const override;

 private:
  Config cfg_{};
};

}  // namespace algconn

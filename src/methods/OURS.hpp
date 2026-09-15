#pragma once
#include <cstddef>

#include "Method.hpp"

namespace algconn {

// Ours: eigensolve-budgeted completed-swap depth search.
//
// PROPOSAL. Effective resistance scores insertion and deletion separately and
// discards their interaction. Ours ranks a completed add/remove pair by the
// exact relative change in spanning-tree volume,
//
//   R(add) - R(remove) - R(add)R(remove) + T(add,remove)^2,
//
// where T is transfer resistance. Let A and R be the absent and present edge
// counts and q=n-1, the rank of a connected Laplacian. A one-swap climb scans
// the union of the q strongest resistance additions crossed with every
// removal and every addition crossed with the q weakest resistance removals.
// Its exact cardinality is
//
//   min(q,A)R + A min(q,R) - min(q,A)min(q,R) <= q*binom(n,2).
//
// Multiplicity-radius roots retain the bidirectional best-response closure of
// that same union. No ER/FV completed-swap rule, fitted top-k, or beam width is
// injected.
// The tree-volume score proposes only; it is not claimed to predict or bound
// the completed swap's change in lambda_2.
//
// DECISION. One paid full eigensystem supplies an a-posteriori residual. A
// shared shifted resolvent and Haynsworth inertia then decide each complete
// rank-two swap in O(1) after an O(n^3) build. A rank-p composite decision costs
// O(n p^2+p^3). Every threshold follows from the measured residual, machine
// precision, or a strict next-representable-number operation.
//
// DEPTH. At a one-swap stall with resolved lambda_2 multiplicity at most two,
// the method pays for one not-yet-tried connected branch having the least
// resolved dip, tries a complete second-swap recovery, and otherwise preserves
// that paid prefix for the next climb. Tried roots persist across restarts. If
// the resolved cluster is greater than two, the only composite radius tried at
// that incumbent is its cluster size: the conservative floating-point
// replacement for the exact interlacing radius. Its root follows the
// coupled response closure. Every later level maximizes the coupled score over
// the complete row of the strongest remaining addition and complete column of
// the weakest remaining removal, without reusing anchor coordinates; only the
// full r-add/r-remove graph is accepted. One depth visit precedes each breadth
// restart, and the best evaluated graph is returned.
//
// B counts actual vector-producing eigensolver invocations. The production
// method fixes the proposal width at n-1 and uses both coupling and inertia.
// Config exposes controlled variants for ablation experiments; its defaults
// are the production method. In the standard fixed-precision word-RAM model,
// every path between
// two paid solves is O(n^3): exact bridge preprocessing makes connectivity
// queries O(1), the rank-matched fibers are traversed directly rather than
// through a skipped O(n^4) Cartesian shell, composite coordinate work is at most
// (n-2)binom(n,2), rank updates never refactor, and the final full-edit inertia
// calculation is O(n^3). Numerical refinement and symmetric QL both have
// machine-derived finite iteration bounds. Thus a budget-B run is O(B n^3)
// arithmetic and contains no data-unbounded inner loop.
class OURS : public Method {
 public:
  struct Config {
    // Ablation hooks. Defaults give the production method.
    // attack=false removes depth; restart=false removes breadth.
    bool attack = true;
    bool restart = true;
    // false replaces the coupled tree-volume score by R_add-R_remove while
    // preserving the candidate set and every other search rule.
    bool includeCoupling = true;
    // false pays a full eigensolve for connected proposals in score order.
    // This clean comparison is defined only with attack=false.
    bool inertiaScreen = true;
    // 0 selects the production width n-1; positive values are sensitivity
    // variants and are capped by the available additions/removals.
    int proposalWidth = 0;
  };

  OURS() = default;
  explicit OURS(Config cfg) : cfg_(cfg) {}
  std::string name() const override { return "ours"; }
  Result run(Graph start, Oracle& oracle, Rng& rng) const override;

  // Public so the structural bound, uniqueness, union-max inclusion, and
  // ordering of the actual proposal oracle can be checked directly.
  static std::vector<Swap> coupledProposals(const Graph& g);
  static size_t coupledProposalPairsExamined(const Graph& g);

  // Structural work certificate for the production multiplicity-radius
  // continuation. `radius` is the resolved lambda_2 multiplicity in run(),
  // not a user setting. After the root, level l has at most P-2l eligible
  // coordinate pairs, so the complete path examines (r-1)(P-r) pairs.
  static size_t cubicContinuationPairBound(const Graph& g, int radius);

 private:
  Config cfg_{};
};

}  // namespace algconn

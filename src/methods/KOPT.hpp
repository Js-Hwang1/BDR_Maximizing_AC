#pragma once
#include <string>
#include "Method.hpp"

namespace algconn {

// Greedy k-opt exchange over the augment set (port of `kopt_exchange_complete`
// from the CDC 2026 companion code, Zhu).
//
// A connected base (here: a BFS spanning tree of the start graph) is
// immutable; the remaining q = m - n + 1 "augment" pairs are exchanged. Each
// round: rank absent pairs by the Fiedler gap of the CURRENT graph (free
// screening from the call already made) and keep the top `pool`; for each
// add-combination of size k, eigensolve the augmented graph (m + k edges,
// counted off-manifold) to rank the SELECTED pairs ascending, and try
// remove-combinations of size k from the bottom `pool`; accept the best
// strictly improving exchange, else stop.
//
// Two deliberate departures from the reference, both protocol-driven:
//   * the initial augment set is the start graph's non-tree pairs, so the
//     method literally starts from the shared per-seed G0 (the reference
//     seeds by sequential Fiedler greedy from its base);
//   * every exact evaluation goes through the budget-counting oracle.
// Connectivity never needs checking: the tree base is always present.
//
// With the default k = 1 and caps >= pool the method is deterministic; the
// generator is used only when k > 1 forces combination sampling.
class KOPT : public Method {
 public:
  struct Config {
    // Somisetty et al. (arXiv:2409.15506) take the pool size m as an input to
    // Algorithm 3, required only to satisfy m >= k and described as
    // "adjustable based on computational needs". Their experiments use
    // m = 20 throughout (1-opt and 2-opt at m = 20, and the MAC comparison at
    // m = 20), with m = 30 in one benchmark, and evaluate k in {1, 2, 3}.
    //
    // k = 1 at m = 20 is the paper's 1-opt. It costs 20 + 20*20 = 420
    // evaluations per round, so a budget buys many rounds. k = 2 at m = 20
    // costs C(20,2)*(1 + C(20,2)) = 36,290 per round, which exceeds the whole
    // budget in the sparse-n experiments and would leave the method stranded
    // mid-round. k is exposed for the larger budgets where it pays.
    int k = 1;            // the paper's 1-opt
    int pool = 20;        // the paper's `m`
    int comboCapAdd = 500;
    int comboCapDel = 500;
    // The paper terminates when no improving k-exchange exists, NOT after a
    // fixed number of rounds: "this iterative process continues until no
    // further improvements in algebraic connectivity can be made" (Sec. V-B).
    // max_rounds=20 is an artifact of the companion code, and under a matched
    // budget it is disqualifying: 420 calls/round x 20 caps the method at
    // ~8.4e3 evaluations however large B is, so it would be plotted as though
    // it had spent B when it structurally cannot. Rounds are bounded by the
    // budget instead, and a converged search restarts, exactly as FV and ER do.
    int maxRounds = 1000000;   // safety only; the budget binds first
    int stagnationLimit = 25;  // consecutive restarts buying no new call
    int maxRestarts = 100000;
  };

  KOPT() = default;
  explicit KOPT(Config cfg) : cfg_(cfg) {}
  // The paper presents 1-opt, 2-opt and 3-opt as distinct heuristics, so k
  // belongs in the name -- they are different curves, not one curve retuned.
  std::string name() const override {
    return cfg_.k <= 1 ? "kopt" : "kopt" + std::to_string(cfg_.k);
  }
  Result run(Graph start, Oracle& oracle, Rng& rng) const override;

 private:
  Config cfg_{};
};

}  // namespace algconn

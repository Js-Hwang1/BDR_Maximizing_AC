#pragma once
#include "Method.hpp"

namespace algconn {

// MDMD: iterated minimum-degree / maximum-distance edge addition.
//
// Port of `mdmd_complete` from the CDC 2026 companion code (Zhu), itself a
// repeated single-edge form of the classical MACP heuristic: at each step,
// (1) pick a minimum-degree vertex i0, tie-broken by minimum extendibility
//     centrality (EC = sum of neighbor degrees), then by index;
// (2) BFS from i0 and collect its farthest non-neighbors;
// (3) pick j0 among them by the same tie-breaking;
// (4) add edge (i0, j0).
//
// Adaptation to this paper's protocol: the immutable base is a BFS spanning
// tree of the start graph, completed by MDMD to m edges. The construction is
// deterministic and consumes ZERO eigensolves; the single oracle call is the
// final evaluation. Under Problem 2 this is a B = 1 constructor -- the
// cheapest nontrivial point on the budget axis, sitting between the
// closed-form families (B = 0) and any search.
class MDMD : public Method {
 public:
  struct Config {
    bool useEc = true;  // extendibility-centrality tie-breaking (reference default)
  };

  MDMD() = default;
  explicit MDMD(Config cfg) : cfg_(cfg) {}
  std::string name() const override { return "mdmd"; }
  Result run(Graph start, Oracle& oracle, Rng& rng) const override;

 private:
  Config cfg_{};
};

}  // namespace algconn

#pragma once
#include "Method.hpp"

namespace algconn {

// Uniformly random completion -- the mandated floor (CLAUDE.md Sec. 11).
//
// Port of `random_complete` from the CDC 2026 companion code (Zhu): repeatedly
// pick an absent pair uniformly at random and add it, until m edges. The base
// is a BFS spanning tree of the start graph, so the result is connected.
//
// Budget: B = 1, the final evaluation. This is the reference against which any
// method's improvement is measured; a method that cannot beat it is not doing
// anything.
class RANDOM : public Method {
 public:
  RANDOM() = default;
  std::string name() const override { return "random"; }
  Result run(Graph start, Oracle& oracle, Rng& rng) const override;
};

}  // namespace algconn

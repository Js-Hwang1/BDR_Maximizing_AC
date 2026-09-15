#pragma once
#include "Method.hpp"

namespace algconn {

// IMPLEMENTATION DETAIL of FV, not a reported method. FV is a sequence of these
// climbs with restarts between them; this is one climb. It lives under detail/
// so it cannot be mistaken for a baseline in its own right.

// Fiedler-vector greedy swap climb (Ghosh--Boyd, adapted to fixed m).
//
// One step:
//   1. one oracle call yields (lambda_2, v2) of the current graph;
//   2. score EVERY pair by the Fiedler gap (v2_u - v2_v)^2. This is the
//      first-order change in lambda_2 from inserting/deleting that pair, and
//      it is FREE -- the paper's oracle model grants v2 with the call, so
//      screening all O(n^2) candidates costs nothing;
//   3. propose adding the highest-gap non-edge and deleting the lowest-gap
//      edge, excluding bridges so the swap is connectivity-preserving;
//   4. one further oracle call verifies the proposal. Accept iff lambda_2
//      strictly increases; otherwise the graph is a leaf and the climb stops.
//
// Budget: about one call per step, since the accepting call also supplies the
// next step's v2. This is exactly the behaviour the "first-order screening is
// free" convention is designed to reward.
class FVClimb : public Method {
 public:
  struct Config {
    int maxSteps = 100000;  // safety cap; the climb normally halts at a leaf
    int candidates = 8;     // try this many (add, remove) pairs per step
  };

  FVClimb() = default;
  explicit FVClimb(Config cfg) : cfg_(cfg) {}
  std::string name() const override { return "fv"; }
  Result run(Graph start, Oracle& oracle, Rng& rng) const override;

 private:
  Config cfg_{};
};

}  // namespace algconn

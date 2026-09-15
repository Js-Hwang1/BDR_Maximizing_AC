#pragma once
#include <vector>

#include "Method.hpp"

namespace algconn {

// Genetic algorithm over G_{n,m}, matching the reference implementation in
// Js-Hwang1/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning,
// branch NEW, src/ga_main.c (+ crl_ga.md), with the learned policy removed so
// the scoring is the pure Fiedler gap.
//
// The two defining operators are both FIEDLER-GUIDED:
//
//   CROSSOVER  child = union(pa, pb), which has more than m edges; take the
//              Fiedler vector OF THE UNION, score every union edge by the gap
//              (v2_i - v2_j)^2, and delete the lowest-gap NON-BRIDGE edges
//              until exactly m remain. The union's Fiedler vector is the point
//              of the design: it reflects the combined connectivity pattern of
//              both parents, so pruning by it keeps edges that matter in the
//              merged context rather than in either parent alone.
//
//   MUTATION   FV-guided softmax swap, repeated numMutations times. ADD a
//              non-edge sampled proportional to exp(gap/tau); REMOVE a
//              non-bridge edge sampled proportional to exp((maxGap-gap)/tau).
//              Add first, then remove. The Fiedler vector is refreshed every
//              refreshEvery swaps, not every swap.
//
// There is deliberately NO separate local-search/memetic step: the Fiedler
// guidance lives inside the operators.
class GA : public Method {
 public:
  struct Config {
    // Defaults from crl_ga.md ("Hyperparameters"), which is the configuration
    // the reference's reported 109.2% result was produced with:
    //   --pop 20  --gen 50  --mutations 2  --mut-rate 0.8  --elite 0.1
    // NOTE these are CONSTANTS, not n-scaled. ga_main.c's internal fallbacks
    // when the flags are omitted (pop=n, gen=3n, mutations=n/2) are a different
    // configuration; using n/2 mutations is ~6x over-mutation at n=24 and
    // destroys good individuals.
    int popSize = 20;
    int generations = 50;
    int numMutations = 2;
    double popMult = 0.0;   // >0 => popSize = popMult * n
    double genMult = 0.0;   // >0 => generations = genMult * n
    double mutMult = 0.0;   // >0 => numMutations = mutMult * n
    // Quadratic scaling, tested because the feasible set grows as
    // 2^Theta(n^2): >0 => the quantity is scaled by n^2 instead of n.
    double popQuad = 0.0;
    double genQuad = 0.0;

    double mutationRate = 0.8;
    double eliteFrac = 0.1;
    int tournament = 3;
    double tau = 0.1;        // softmax temperature for the FV-guided sampling
    int refreshEvery = 0;    // 0 => n/4, min 2

    // Ablation switch matching `use_random_mut` in the reference: uniform
    // sampling instead of Fiedler-guided.
    bool randomMutation = false;

    // Stop when no NEW oracle call has occurred for this many generations. A
    // converged population evaluates entirely from cache, so budget exhaustion
    // alone never fires.
    int stagnationLimit = 40;
  };

  GA() = default;
  explicit GA(Config cfg) : cfg_(cfg) {}
  std::string name() const override { return "ga"; }
  Result run(Graph start, Oracle& oracle, Rng& rng) const override;

  // Union-then-prune crossover. Exposed for testing.
  static Graph crossoverUnionPrune(const Graph& pa, const Graph& pb,
                                   Oracle& oracle, Rng& rng);

  // Move `g` k swaps along a swap-graph geodesic toward `target`. Retained as a
  // tested utility (it realises the swap-distance construction); no longer used
  // by crossover.
  static int rewireToward(Graph& g, const Graph& target, int k, Rng& rng);

 private:
  Config cfg_{};
};

}  // namespace algconn

#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "Graph.hpp"

namespace algconn {

// Full symmetric eigensolves, counted.
//
// One unit of B is one actual invocation of the full vector-producing
// eigensolver. Every first evaluation therefore stores lambda_2, lambda_3,
// multiplicity, v2, and v3. Reusing those stored quantities costs no solve.
// A full n-by-n eigenbasis is intentionally not retained: if a caller discards
// it and requests it again, the recomputation is another physical eigensolve
// and consumes another unit. Thus calls() == realSolves() by construction;
// neither memoization nor a lazy vector upgrade can hide work from B.
class Oracle {
 public:
  struct Eval {
    double lam2 = 0.0;
    // lambda_3, i.e. the eigenvalue immediately above lambda_2. The solver
    // always produces the full spectrum, so exposing it costs nothing and
    // stays inside the oracle model (one call returns the spectrum). It is
    // what the interlacing escape test needs: a single further swap from this
    // graph cannot push lambda_2 above lambda_3 of this graph.
    double lam3 = 0.0;
    int multiplicity = 0;    // multiplicity of lambda_2; > 1 is normal at optima
    std::vector<double> v2;
    // Eigenvector of lambda_3. Together with v2 it forms the supergradient of
    // the Ky Fan 2-sum
    // lambda_2 + lambda_3 -- the degenerate-safe first-order object a method
    // needs exactly where the lone Fiedler vector is ill-posed.
    std::vector<double> v3;
  };

  explicit Oracle(long budget = -1) : budget_(budget) {}

  // Cached lambda_2. The first request performs one full eigensolve.
  double lam2(const Graph& g);
  // Cached lambda_2 with v2; the first request performs one full eigensolve.
  const Eval& eigenpair(const Graph& g);
  // multiplicity of lambda_2 at g (no extra charge if already evaluated).
  int multiplicity(const Graph& g);
  // lambda_3 at g (no extra charge if already evaluated; 0 if disconnected).
  double lam3(const Graph& g);
  // Full eigendecomposition of g's Laplacian: vals ascending, vecs row-major
  // n*n with column k the eigenvector of vals[k]. The n-by-n basis is not
  // cached, so EVERY invocation performs and charges one actual eigensolve.
  // Throws if g is disconnected (callers screen connectivity for free first).
  void eigensystem(const Graph& g, std::vector<double>& vals,
                   std::vector<double>& vecs);

  long calls() const { return calls_; }
  // Auditable invariant: every actual eigensolve consumes exactly one call.
  long realSolves() const { return solvesVectors_; }
  long solvesValues() const { return 0; }
  long solvesVectors() const { return solvesVectors_; }
  long upgrades() const { return 0; }
  long callsOffManifold() const { return offManifold_; }
  long cacheHits() const { return cacheHits_; }
  // Calls spent on graphs that are NOT candidate solutions -- i.e. whose edge
  // count differs from the first graph evaluated. The GA's crossover
  // eigensolves the UNION of two parents, which has more than m edges, so part
  // of its budget buys guidance rather than evaluating a feasible graph. This
  // makes that asymmetry measurable instead of hidden.
  int referenceEdges() const { return refM_; }
  long queries() const { return queries_; }
  long budget() const { return budget_; }
  bool exhausted() const { return budget_ >= 0 && calls_ >= budget_; }
  long remaining() const { return budget_ < 0 ? -1 : budget_ - calls_; }

  // Running best over everything ever evaluated, and the budget at which each
  // new best appeared. One run therefore yields the entire quality-vs-B curve,
  // which is the paper's central figure, at no extra cost.
  double best() const { return best_; }
  const std::vector<std::pair<long, double>>& trace() const { return trace_; }

  void reset() {
    cache_.clear();
    lastFullKey_.clear();
    lastFullEval_ = Eval{};
    hasLastFull_ = false;
    trace_.clear();
    calls_ = queries_ = solvesVectors_ = offManifold_ = cacheHits_ = 0;
    refM_ = -1;
    best_ = -1.0;
  }

 private:
  struct KeyHash {
    size_t operator()(const std::vector<uint64_t>& k) const noexcept {
      size_t h = 1469598103934665603ULL;
      for (uint64_t w : k) { h ^= w; h *= 1099511628211ULL; }
      return h;
    }
  };
  Eval& compute(const Graph& g);

  long budget_;
  long calls_ = 0;
  long queries_ = 0;
  double best_ = -1.0;
  long solvesVectors_ = 0;
  long offManifold_ = 0;
  long cacheHits_ = 0;
  int refM_ = -1;
  std::vector<std::pair<long, double>> trace_;
  // eigensystem() always pays, so OURS needs no history table. Retain only its
  // latest metadata to honor a subsequent cached lam2/eigenpair request while
  // keeping the paid full-solve path deterministic O(n^3), independent of
  // hash-table collision behavior.
  std::vector<uint64_t> lastFullKey_;
  Eval lastFullEval_;
  bool hasLastFull_ = false;
  std::unordered_map<std::vector<uint64_t>, Eval, KeyHash> cache_;
};

}  // namespace algconn

// Self-contained test suite. No framework: a failure aborts with a message.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "Eigen.hpp"
#include "Graph.hpp"
#include "MomentBounds.hpp"
#include "Oracle.hpp"
#include "methods/detail/FVClimb.hpp"
#include "methods/FV.hpp"
#include "methods/GA.hpp"
#include "methods/MDMD.hpp"
#include "methods/KOPT.hpp"
#include "methods/OURS.hpp"
#include "methods/RANDOM.hpp"
#include "methods/ER.hpp"
#include "ParallelRestart.hpp"
#include "methods/detail/GroundedInverse.hpp"
#include "methods/detail/TreeBase.hpp"

using namespace algconn;

static int g_checks = 0, g_fail = 0;

static void expectTrue(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}
static void expectNear(double got, double want, double tol, const std::string& what) {
  ++g_checks;
  if (!(std::fabs(got - want) <= tol)) {
    ++g_fail;
    std::printf("  FAIL  %s: got %.12g want %.12g (tol %g)\n", what.c_str(), got, want, tol);
  }
}

// ---- helpers ---------------------------------------------------------------
static Graph fromEdges(const PairIndex& pix, const std::vector<std::pair<int,int>>& es) {
  Bitset b(pix.numPairs());
  for (auto [u, v] : es) b.set(pix.index(u, v));
  return Graph(pix, std::move(b));
}
static double lam2Of(const Graph& g) {
  std::vector<double> a = g.laplacian(), vals;
  eig::symmetric(a, g.n(), vals, false);
  return vals[1];
}

// ---- 1. eigensolver against closed forms ----------------------------------
static void testEigenClosedForms() {
  std::printf("eigensolver vs closed forms\n");
  const double PI = 3.14159265358979323846;
  for (int n : {3, 5, 8, 13, 24, 40}) {
    PairIndex pix(n);

    std::vector<std::pair<int,int>> K;
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) K.push_back({i, j});
    expectNear(lam2Of(fromEdges(pix, K)), n, 1e-9, "lambda2(K_n) = n, n=" + std::to_string(n));

    std::vector<std::pair<int,int>> C;
    for (int i = 0; i < n; ++i) C.push_back({std::min(i,(i+1)%n), std::max(i,(i+1)%n)});
    expectNear(lam2Of(fromEdges(pix, C)), 2 * (1 - std::cos(2 * PI / n)), 1e-9,
               "lambda2(C_n), n=" + std::to_string(n));

    std::vector<std::pair<int,int>> P;
    for (int i = 0; i + 1 < n; ++i) P.push_back({i, i + 1});
    expectNear(lam2Of(fromEdges(pix, P)), 2 * (1 - std::cos(PI / n)), 1e-9,
               "lambda2(P_n), n=" + std::to_string(n));

    std::vector<std::pair<int,int>> S;
    for (int j = 1; j < n; ++j) S.push_back({0, j});
    expectNear(lam2Of(fromEdges(pix, S)), 1.0, 1e-9,
               "lambda2(star_n) = 1, n=" + std::to_string(n));
  }
  // complete bipartite K_{a,b}: lambda2 = min(a,b)
  for (auto [a, b] : std::vector<std::pair<int,int>>{{2,3},{3,3},{4,9},{6,6},{5,12}}) {
    PairIndex pix(a + b);
    std::vector<std::pair<int,int>> E;
    for (int i = 0; i < a; ++i) for (int j = 0; j < b; ++j) E.push_back({i, a + j});
    expectNear(lam2Of(fromEdges(pix, E)), std::min(a, b), 1e-9,
               "lambda2(K_{a,b}) = min(a,b)");
  }
}

// ---- 1b. the fast path must agree exactly with the validated one ----------
static void testValuesOnlyPathAgrees() {
  std::printf("eigenvalues-only fast path\n");
  Rng rng(31337);
  for (int n : {5, 9, 16, 24, 32}) {
    PairIndex pix(n);
    for (int m : {n, 2 * n, pix.numPairs() / 2, pix.numPairs() - 1}) {
      if (m < n - 1 || m > pix.numPairs()) continue;
      for (int t = 0; t < 12; ++t) {
        Graph g = Graph::randomConnected(pix, m, rng);
        std::vector<double> a1 = g.laplacian(), v1;
        std::vector<double> a2 = g.laplacian(), v2;
        eig::symmetric(a1, n, v1, true);
        eig::symmetricValues(a2, n, v2);
        double worst = 0.0;
        for (int k = 0; k < n; ++k) worst = std::max(worst, std::fabs(v1[k] - v2[k]));
        expectTrue(worst < 1e-11, "values-only path matches the vector path");
        expectTrue(v2[0] < 1e-9, "lambda_1 = 0 for a connected Laplacian");
        expectTrue(v2[1] > -1e-12, "lambda_2 >= 0");
      }
    }
  }
}

// ---- 1c. multiplicity of lambda_2 -----------------------------------------
static void testMultiplicity() {
  std::printf("lambda_2 multiplicity\n");
  const double PI = 3.14159265358979323846;
  for (int n : {6, 9, 12, 20}) {
    PairIndex pix(n);
    // K_n: lambda_2 = n with multiplicity n-1 -- the extreme degenerate case
    std::vector<std::pair<int,int>> K;
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) K.push_back({i, j});
    Graph kn = fromEdges(pix, K);
    std::vector<double> a = kn.laplacian(), v;
    eig::symmetric(a, n, v, false);
    expectTrue(eig::lambda2Multiplicity(v) == n - 1, "mult(K_n) = n-1");

    // C_n: lambda_2 = 2(1-cos(2pi/n)) with multiplicity 2
    std::vector<std::pair<int,int>> C;
    for (int i = 0; i < n; ++i) C.push_back({std::min(i,(i+1)%n), std::max(i,(i+1)%n)});
    Graph cn = fromEdges(pix, C);
    a = cn.laplacian();
    eig::symmetric(a, n, v, false);
    expectTrue(eig::lambda2Multiplicity(v) == 2, "mult(C_n) = 2");
    (void)PI;

    // path: lambda_2 is simple
    std::vector<std::pair<int,int>> P;
    for (int i = 0; i + 1 < n; ++i) P.push_back({i, i + 1});
    Graph pn = fromEdges(pix, P);
    a = pn.laplacian();
    eig::symmetric(a, n, v, false);
    expectTrue(eig::lambda2Multiplicity(v) == 1, "mult(P_n) = 1");
  }
}

// ---- 2. (n,m) invariant under every operator -------------------------------
static void testInvariants() {
  std::printf("(n,m) invariant\n");
  Rng rng(12345);
  for (int n : {6, 10, 16, 24}) {
    PairIndex pix(n);
    const int np = pix.numPairs();
    for (int m : {n - 1, n, n + 5, np / 2, np - 1, np}) {
      if (m < n - 1 || m > np) continue;
      for (int t = 0; t < 20; ++t) {
        Graph g = Graph::randomConnected(pix, m, rng);
        expectTrue(g.m() == m, "randomConnected edge count");
        expectTrue(g.edges().count() == m, "bitset popcount == m");
        bool ok = true;
        try { g.checkFeasible(); } catch (...) { ok = false; }
        expectTrue(ok, "randomConnected feasible");

        // apply many random legal swaps; m must never move
        for (int s = 0; s < 50; ++s) {
          std::vector<int> present = g.edges().ones(), absent;
          for (int k = 0; k < np; ++k) if (!g.has(k)) absent.push_back(k);
          if (present.empty() || absent.empty()) break;
          Swap sw{absent[rng.below((uint32_t)absent.size())],
                  present[rng.below((uint32_t)present.size())]};
          Graph h = g;
          h.applySwap(sw);
          expectTrue(h.m() == m, "applySwap preserves m");
          expectTrue(h.edges().count() == m, "applySwap preserves popcount");
          if (h.isConnected()) { g = h; }
        }
        bool ok2 = true;
        try { g.checkFeasible(); } catch (...) { ok2 = false; }
        expectTrue(ok2, "graph feasible after 50 swaps");
      }
    }
  }
}

// ---- 3. incremental degree / sumDegSq caches -------------------------------
static void testDegreeCaches() {
  std::printf("incremental degree caches\n");
  Rng rng(99);
  PairIndex pix(18);
  Graph g = Graph::randomConnected(pix, 60, rng);
  for (int s = 0; s < 300; ++s) {
    std::vector<int> present = g.edges().ones(), absent;
    for (int k = 0; k < pix.numPairs(); ++k) if (!g.has(k)) absent.push_back(k);
    Swap sw{absent[rng.below((uint32_t)absent.size())],
            present[rng.below((uint32_t)present.size())]};
    g.applySwap(sw);
    // checkFeasible recomputes degrees and sumDegSq from scratch and compares
    bool ok = true;
    try { if (g.isConnected()) g.checkFeasible(); } catch (...) { ok = false; }
    expectTrue(ok, "degree/sumDegSq caches stay exact under applySwap");
    if (!ok) break;
  }
}

// ---- 4. moment bound is certified ------------------------------------------
static void testMomentBound() {
  std::printf("zero-cost moment bound\n");
  Rng rng(7);
  for (int n : {6, 10, 16, 24}) {
    PairIndex pix(n);
    for (int m : {n, n + 4, pix.numPairs() / 2, pix.numPairs() - 1}) {
      if (m < n - 1 || m > pix.numPairs()) continue;
      for (int t = 0; t < 40; ++t) {
        Graph g = Graph::randomConnected(pix, m, rng);
        expectTrue(lam2Of(g) <= g.momentUpperBound() + 1e-9,
                   "lambda2 <= momentUpperBound");
      }
    }
  }
}

// ---- 5. bridges: removing one must disconnect ------------------------------
static void testBridges() {
  std::printf("bridge detection\n");
  Rng rng(4242);
  for (int n : {6, 10, 16}) {
    PairIndex pix(n);
    for (int m : {n - 1, n, n + 3}) {
      for (int t = 0; t < 20; ++t) {
        Graph g = Graph::randomConnected(pix, m, rng);
        std::vector<char> isB(pix.numPairs(), 0);
        for (int k : g.bridgePairs()) isB[k] = 1;
        for (int k : g.edges().ones()) {
          Bitset b = g.edges();
          b.clear(k);
          Graph h(pix, std::move(b));
          expectTrue(h.isConnected() == !isB[k],
                     "bridge iff its removal disconnects");
        }
      }
    }
  }
}

static void testSwapConnectivity() {
  std::printf("completed-swap connectivity oracle\n");
  for (int n : {4, 7, 12, 20}) {
    PairIndex pix(n);
    Rng rng(static_cast<uint64_t>(701 + n));
    for (int m : {n - 1, n, 2 * n, pix.numPairs() - 1}) {
      if (m < n - 1 || m >= pix.numPairs()) continue;
      for (int sample = 0; sample < 8; ++sample) {
        const Graph graph = Graph::randomConnected(pix, m, rng);
        const SwapConnectivity connectivity(graph);
        for (int add = 0; add < pix.numPairs(); ++add) {
          if (graph.has(add)) continue;
          for (int remove = 0; remove < pix.numPairs(); ++remove) {
            if (!graph.has(remove)) continue;
            const Swap swap{add, remove};
            expectTrue(connectivity.keepsConnected(swap) ==
                           graph.swapKeepsConnected(swap),
                       "preprocessed connectivity matches direct traversal");
          }
        }
      }
    }
  }
}

// ---- 5b. the optimisation FVClimb relies on --------------------------------
static void testNonBridgeRemovalKeepsConnected() {
  std::printf("non-bridge removal preserves connectivity\n");
  Rng rng(8080);
  for (int n : {8, 14, 20}) {
    PairIndex pix(n);
    for (int m : {n, n + 5, 2 * n, pix.numPairs() / 2}) {
      if (m > pix.numPairs()) continue;
      for (int t = 0; t < 15; ++t) {
        Graph g = Graph::randomConnected(pix, m, rng);
        std::vector<char> isB(pix.numPairs(), 0);
        for (int k : g.bridgePairs()) isB[k] = 1;
        // For EVERY non-bridge removal and EVERY insertion, the swap must keep
        // the graph connected. This is what licenses dropping the check.
        for (int r : g.edges().ones()) {
          if (isB[r]) continue;
          for (int a = 0; a < pix.numPairs(); ++a) {
            if (g.has(a)) continue;
            Graph h = g;
            h.applySwap(Swap{a, r});
            expectTrue(h.isConnected(), "non-bridge swap stays connected");
          }
        }
      }
    }
  }
}

// ---- 5c. degenerate lambda_2 certifies local optimality ---------------------
static void testDegeneracyImpliesLocalOptimum() {
  std::printf("degenerate lambda_2 => local optimum\n");
  // A swap gives L' = L + bb^T - cc^T. Interlacing for the rank-one PSD update
  // gives lambda_2(L + bb^T) <= lambda_3(L); interlacing for the rank-one
  // downdate gives lambda_2(L') <= lambda_2(L + bb^T). Hence
  //     lambda_2(L') <= lambda_3(L),
  // which equals lambda_2(L) exactly when lambda_2 is degenerate. So no swap
  // can strictly improve a graph whose lambda_2 is repeated.
  Rng rng(4711);
  for (int n : {6, 8, 10, 12}) {
    PairIndex pix(n);
    const int np = pix.numPairs();
    for (int m = n; m <= np - 1; m += 2) {
      for (int t = 0; t < 6; ++t) {
        Graph g = Graph::randomConnected(pix, m, rng);
        std::vector<double> a = g.laplacian(), v;
        eig::symmetric(a, n, v, false);
        const double lam2 = v[1], lam3 = v[2];
        const int mult = eig::lambda2Multiplicity(v);
        for (int r = 0; r < np; ++r) {
          if (!g.has(r)) continue;
          for (int ad = 0; ad < np; ++ad) {
            if (g.has(ad)) continue;
            Graph h = g;
            h.applySwap(Swap{ad, r});
            if (!h.isConnected()) continue;
            std::vector<double> a2 = h.laplacian(), v2;
            eig::symmetric(a2, n, v2, false);
            expectTrue(v2[1] <= lam3 + 1e-9, "lambda_2(L') <= lambda_3(L)");
            if (mult >= 2)
              expectTrue(v2[1] <= lam2 + 1e-12,
                         "no swap improves a degenerate lambda_2");
          }
        }
      }
    }
  }
}

// ---- 5d. moment-bound certificates ----------------------------------------
static void testMomentBounds() {
  std::printf("moment bound certificates\n");
  Rng rng(90210);

  // (a) NO graph may exceed the (n,m)-only certificates.
  for (int n : {6, 10, 16, 24}) {
    PairIndex pix(n);
    const int np = pix.numPairs();
    for (int m : {n, n + 3, 2 * n, np / 2, np - 1, np}) {
      if (m < n - 1 || m > np) continue;
      const double bF = mb::fiedler(n, m);
      const double b2 = mb::twoMomentNM(n, m);
      const double b3 = mb::threeMomentNM(n, m);
      for (int t = 0; t < 25; ++t) {
        Graph g = Graph::randomConnected(pix, m, rng);
        std::vector<double> a = g.laplacian(), v;
        eig::symmetric(a, n, v, false);
        const double lam2 = v[1];
        expectTrue(lam2 <= bF + 1e-7, "lambda_2 <= Fiedler bound");
        expectTrue(lam2 <= b2 + 1e-7, "lambda_2 <= two-moment (n,m) bound");
        expectTrue(lam2 <= b3 + 1e-7, "lambda_2 <= three-moment (n,m) bound");
        // per-graph versions, evaluated on the graph's own invariants
        std::vector<int> deg(g.degrees());
        long tri = 0;
        for (int i = 0; i < n; ++i)
          for (int j = i + 1; j < n; ++j) {
            if (!g.has(pix.index(i, j))) continue;
            for (int k2 = j + 1; k2 < n; ++k2)
              if (g.has(pix.index(i, k2)) && g.has(pix.index(j, k2))) ++tri;
          }
        expectTrue(lam2 <= mb::twoMoment(n, m, deg) + 1e-7, "own-degree 2-moment bound");
        expectTrue(lam2 <= mb::threeMoment(n, m, deg, tri) + 1e-7, "own-degree 3-moment bound");
      }
    }
  }

  // (b) EQUALITY: the star is certified exactly by the two-moment bound.
  for (int n : {6, 9, 12, 17, 24, 31}) {
    PairIndex pix(n);
    std::vector<std::pair<int,int>> E;
    for (int j = 1; j < n; ++j) E.push_back({0, j});
    Graph s = fromEdges(pix, E);
    expectNear(mb::twoMoment(n, n - 1, s.degrees()), 1.0, 1e-9,
               "two-moment bound == 1 at the star (equality case)");
    expectNear(lam2Of(s), 1.0, 1e-9, "star lambda_2 == 1");
  }

  // (c) EQUALITY: balanced complete bipartite, EVEN n only.
  for (int n : {6, 8, 12, 16, 24}) {
    PairIndex pix(n);
    const int a = n / 2;
    std::vector<std::pair<int,int>> E;
    for (int i = 0; i < a; ++i) for (int j = 0; j < a; ++j) E.push_back({i, a + j});
    Graph g = fromEdges(pix, E);
    expectNear(mb::twoMoment(n, (long)a * a, g.degrees()), (double)a, 1e-9,
               "two-moment bound == n/2 at K_{n/2,n/2}");
    expectNear(lam2Of(g), (double)a, 1e-9, "K_{n/2,n/2} lambda_2 == n/2");
  }

  // (d) the three-moment bound must never be looser than reality allows, and
  //     must decrease in the triangle count (this is what makes T=0 valid).
  {
    const int n = 16, a = 4, m = a * (n - a);
    std::vector<int> deg;
    for (int i = 0; i < a; ++i) deg.push_back(n - a);
    for (int i = 0; i < n - a; ++i) deg.push_back(a);
    double prev = 1e18;
    for (long T : {0L, 5L, 20L, 80L}) {
      const double b = mb::threeMoment(n, m, deg, T);
      expectTrue(b < prev + 1e-12, "three-moment bound is non-increasing in T");
      prev = b;
    }
  }
}

// ---- 6. oracle budget accounting -------------------------------------------
static void testOracle() {
  std::printf("oracle budget\n");
  Rng rng(1);
  PairIndex pix(12);
  Graph g = Graph::randomConnected(pix, 30, rng);
  Oracle o;
  for (int i = 0; i < 10; ++i) o.lam2(g);
  expectTrue(o.calls() == 1, "repeat evaluations are memoized, charged once");
  expectTrue(o.realSolves() == 1, "one charged call is one actual eigensolve");
  expectTrue(o.cacheHits() == 9,
             "cache-hit counter records every reused graph evaluation");
  expectTrue(o.queries() == 10, "queries counts every request");
  const std::vector<double> cachedV2 = o.eigenpair(g).v2;
  expectTrue((int)cachedV2.size() == g.n(), "the first solve retained v2");
  expectTrue(o.calls() == 1 && o.realSolves() == 1,
             "requesting cached v2 performs no hidden upgrade solve");

  std::vector<double> vals, vecs;
  o.eigensystem(g, vals, vecs);
  expectTrue(o.calls() == 2 && o.realSolves() == 2,
             "recomputing a discarded full basis is charged");
  expectTrue((int)vals.size() == g.n() && (int)vecs.size() == g.n() * g.n(),
             "charged full eigensystem has the promised dimensions");

  o.reset();
  expectTrue(o.calls() == 0 && o.realSolves() == 0 && o.queries() == 0 &&
                 o.cacheHits() == 0,
             "oracle reset clears every accounting counter");
  o.eigensystem(g, vals, vecs);
  expectTrue(o.calls() == 1 && o.realSolves() == 1,
             "a direct full eigensystem is exactly one budget unit");
  (void)o.lam2(g);
  expectTrue(o.calls() == 1 && o.realSolves() == 1,
             "metadata from a full eigensystem is memoized without a solve");

  Oracle b(3);
  Rng r2(2);
  try {
    for (int i = 0; i < 100; ++i)
      b.lam2(Graph::randomConnected(pix, 30, r2));
  } catch (const std::exception&) {}
  expectTrue(b.calls() <= 3, "oracle never exceeds its budget");
  expectTrue(b.calls() == b.realSolves(),
             "budget calls equal physical eigensolves at exhaustion");
}

// ---- 7. FV-greedy is a rewire and never decreases lambda_2 -----------------
static void testFvClimb() {
  std::printf("FV-greedy rewire\n");
  for (int n : {10, 16, 20}) {
    PairIndex pix(n);
    for (int m : {n + 2, 2 * n, 3 * n}) {
      if (m > pix.numPairs()) continue;
      Rng rng(n * 100 + m);
      Graph start = Graph::randomConnected(pix, m, rng);
      const double before = lam2Of(start);
      Oracle o;
      Result r = FVClimb().run(start, o, rng);
      expectTrue((int)r.edges.size() == m, "FV-greedy returns exactly m edges");
      expectTrue(r.lam2 >= before - 1e-12, "FV-greedy never decreases lambda_2");
      expectNear(r.lam2, lam2Of(fromEdges(pix, r.edges)), 1e-9,
                 "reported lambda_2 matches the returned graph");
      expectTrue(r.lam2 <= fromEdges(pix, r.edges).momentUpperBound() + 1e-9,
                 "FV-greedy result respects the moment bound");
    }
  }
}


// ---- 8. GA crossover is a genuine rewiring path ----------------------------
static void testCrossoverIsRewire() {
  std::printf("GA crossover = geodesic rewiring\n");
  Rng rng(2024);
  for (int n : {8, 12, 20, 24}) {
    PairIndex pix(n);
    for (int m : {n + 2, 2 * n, pix.numPairs() / 2}) {
      if (m > pix.numPairs()) continue;
      for (int t = 0; t < 25; ++t) {
        Graph a = Graph::randomConnected(pix, m, rng);
        Graph b = Graph::randomConnected(pix, m, rng);
        const int d = (int)a.edges().minus(b.edges()).size();

        // k = 0 is the identity
        Graph c0 = a;
        expectTrue(GA::rewireToward(c0, b, 0, rng) == 0, "k=0 applies no swap");
        expectTrue(c0.edges() == a.edges(), "k=0 returns the parent unchanged");

        // k = d reaches the other parent exactly (swap distance, Prop. 1)
        Graph cd = a;
        const int used = GA::rewireToward(cd, b, d, rng);
        expectTrue(used == d, "k=d applies exactly d swaps");
        expectTrue(cd.edges() == b.edges(), "k=d reaches the other parent");

        // intermediate k: feasible, and exactly k closer to b
        for (int k : {1, d / 3, d / 2}) {
          if (k <= 0 || k > d) continue;
          Graph c = a;
          const int u = GA::rewireToward(c, b, k, rng);
          expectTrue(c.m() == m, "crossover preserves m");
          expectTrue(c.edges().count() == m, "crossover preserves popcount");
          bool ok = true;
          try { c.checkFeasible(); } catch (...) { ok = false; }
          expectTrue(ok, "crossover child is feasible and connected");
          const int dist = (int)c.edges().minus(b.edges()).size();
          expectTrue(dist == d - u, "child lies on a geodesic toward the parent");
        }
      }
    }
  }
}

// ---- 9. GA end to end -------------------------------------------------------
static void testGa() {
  std::printf("GA end to end\n");
  for (int n : {10, 16, 24}) {
    PairIndex pix(n);
    for (int m : {n + 3, 2 * n}) {
      if (m > pix.numPairs()) continue;
      Rng rng(n * 77 + m);
      Graph start = Graph::randomConnected(pix, m, rng);
      Oracle o;
      GA::Config cfg; cfg.popSize = 20; cfg.generations = 15;
      Result r = GA(cfg).run(start, o, rng);
      expectTrue((int)r.edges.size() == m, "GA returns exactly m edges");
      Graph got = fromEdges(pix, r.edges);
      expectTrue(got.m() == m, "GA result has m edges when rebuilt");
      expectNear(r.lam2, lam2Of(got), 1e-9, "GA lambda_2 matches its graph");
      expectTrue(got.isConnected(), "GA result is connected");
      expectTrue(r.lam2 <= got.momentUpperBound() + 1e-9, "GA respects moment bound");
      // With memetic climbs a generation costs MORE than one call per
      // individual, so the old population x generations bound no longer holds.
      // What must hold: the run terminates, and it terminates because of
      // stagnation or the generation cap rather than by spinning.
      expectTrue(r.steps <= cfg.generations, "GA respects its generation cap");
      expectTrue(r.oracleCalls > 0, "GA actually consulted the oracle");
    }
  }
  // A converged population evaluates entirely from cache and must still
  // terminate: this is the bug that stagnationLimit exists to prevent.
  {
    PairIndex pix2(10);
    Rng rng2(3);
    Graph s = Graph::randomConnected(pix2, 40, rng2);
    Oracle o;  // unlimited: only stagnation can stop this
    GA::Config c; c.popSize = 4; c.generations = 300; c.stagnationLimit = 20;
    const Result r = GA(c).run(s, o, rng2);
    // With FV-guided softmax mutation the population keeps producing genuinely
    // new graphs, so stagnation need not fire; what must hold is that the run
    // TERMINATES and stays feasible. Real cache-only convergence is covered by
    // the near-complete regression below, where the feasible set is exhausted.
    expectTrue(r.steps <= c.generations, "run terminates within the cap");
    expectTrue((int)r.edges.size() == 40, "run returns m edges");
  }

  // Regression: near m = binom(n,2) the feasible set is tiny (at m = N-1 there
  // are only N graphs), so every evaluation is soon a cache hit and the budget
  // can NEVER be exhausted. Both methods must still terminate.
  for (int n : {10, 12}) {
    PairIndex px(n);
    const int N = px.numPairs();
    for (int m : {N - 1, N - 2, N - 3}) {
      Rng r(n * 17 + m);
      Graph s = Graph::randomConnected(px, m, r);
      { Oracle o(100000);  // unreachable budget by construction
        const Result res = FV().run(s, o, r);
        expectTrue((int)res.edges.size() == m, "FV terminates near-complete");
      }
      { Oracle o(100000);
        GA::Config c; c.popSize = 6;
        const Result res = GA(c).run(s, o, r);
        expectTrue((int)res.edges.size() == m, "GA terminates near-complete"); }
    }
  }

  // budget is a hard stop
  PairIndex pix(14);
  Rng rng(5);
  Graph start = Graph::randomConnected(pix, 40, rng);
  for (long B : {5L, 37L, 200L}) {
    Oracle o(B);
    GA::Config cfg; cfg.popSize = 12; cfg.generations = 100;
    Result r = GA(cfg).run(start, o, rng);
    expectTrue(o.calls() <= B, "GA never exceeds its oracle budget");
    expectTrue((int)r.edges.size() == 40, "budget stop still returns m edges");
  }
}

// ---- 9a. crossover pruning == naive reverse-delete -------------------------
static void testCrossoverPruneEqualsReverseDelete() {
  std::printf("union-prune == reverse-delete\n");
  Rng rng(1234567);
  for (int n : {8, 12, 16}) {
    PairIndex pix(n);
    for (int m : {n + 2, 2 * n, pix.numPairs() / 3}) {
      if (m < n || m >= pix.numPairs()) continue;
      for (int t = 0; t < 8; ++t) {
        Graph a = Graph::randomPathSeeded(pix, m, rng);
        Graph b = Graph::randomPathSeeded(pix, m, rng);
        Oracle o;
        const Graph fast = GA::crossoverUnionPrune(a, b, o, rng);
        expectTrue(fast.m() == m, "crossover yields exactly m edges");
        expectTrue(fast.isConnected(), "crossover child is connected");

        Bitset u(pix.numPairs());
        for (int k : a.edges().ones()) u.set(k);
        for (int k : b.edges().ones()) u.set(k);
        Graph naive(pix, std::move(u));
        if (naive.m() <= m) continue;
        Oracle o2;
        const std::vector<double> v2 = o2.eigenpair(naive).v2;
        std::vector<std::pair<double,int> > sc;
        for (int k : naive.edges().ones()) {
          const double d = v2[pix.i(k)] - v2[pix.j(k)];
          sc.push_back(std::make_pair(d * d, k));
        }
        std::sort(sc.begin(), sc.end(), [](const std::pair<double,int>& x,
                                           const std::pair<double,int>& y) {
          return x.first != y.first ? x.first < y.first : x.second < y.second; });
        int toRemove = naive.m() - m;
        while (toRemove > 0) {
          std::vector<char> isB(pix.numPairs(), 0);
          for (int q : naive.bridgePairs()) isB[q] = 1;
          bool did = false;
          for (size_t z = 0; z < sc.size(); ++z) {
            const int k = sc[z].second;
            if (!naive.has(k) || isB[k]) continue;
            Bitset nb = naive.edges();
            nb.clear(k);
            naive = Graph(pix, std::move(nb));
            --toRemove; did = true; break;
          }
          if (!did) break;
        }
        expectTrue(naive.edges() == fast.edges(),
                   "Kruskal pruning matches reverse-delete exactly");
      }
    }
  }
}

// ---- 9b. GA should beat FV-greedy ON AVERAGE -------------------------------
static void testGaBeatsFvClimbOnAverage() {
  std::printf("GA vs fv_greedy (aggregate)\n");
  // The reference GA has no local-search step, so unlike a memetic variant it
  // does NOT contain FV-greedy and cannot dominate it instance-by-instance.
  // What must hold is that it wins on average.
  double sumGa = 0, sumFv = 0;
  int cells = 0;
  for (int n : {12, 18, 24}) {
    PairIndex pix(n);
    for (int m : {n + 2, 2 * n, 4 * n}) {
      if (m < n || m >= pix.numPairs()) continue;
      for (int s = 0; s < 4; ++s) {
        Rng r1(n * 613 + m * 7 + s), r2(n * 613 + m * 7 + s);
        Graph start = Graph::randomPathSeeded(pix, m, r1);
        Graph start2 = Graph::randomPathSeeded(pix, m, r2);
        Oracle o1;
        sumFv += FVClimb().run(start, o1, r1).lam2;
        Oracle o2(20000);
        sumGa += GA().run(start2, o2, r2).lam2;
        ++cells;
      }
    }
  }
  std::printf("   mean fv_greedy %.5f   mean GA %.5f over %d cells\n",
              sumFv / cells, sumGa / cells, cells);
  expectTrue(sumGa > sumFv, "GA beats FV-greedy on average");
}

// ---- 9c. tournament selection must actually select ------------------------
static void testSelectionPressure() {
  std::printf("selection pressure\n");
  // With tournament size 1 the GA is a random walk with elitism; with size 5 it
  // is strongly directed. At equal budget the directed one must do better on
  // average, otherwise `pick` is not reading fitness at all.
  double weak = 0.0, strong = 0.0;
  const int trials = 24;
  for (int s = 0; s < trials; ++s) {
    PairIndex pix(20);
    for (int which = 0; which < 2; ++which) {
      Rng rng(9090 + s);
      Graph st = Graph::randomConnected(pix, 60, rng);
      Oracle o(1500);
      GA::Config c;
      c.popSize = 8;
      c.eliteFrac = 0.15;
      c.tournament = which ? 5 : 1;
      c.randomMutation = true;  // isolate selection from FV guidance
      const double v = GA(c).run(st, o, rng).lam2;
      (which ? strong : weak) += v / trials;
    }
  }
  expectTrue(strong > weak,
             "tournament size 5 must beat size 1 -- else selection is inert");
  std::printf("   tournament 1: %.5f   tournament 5: %.5f\n", weak, strong);
}

// ---- 10. determinism --------------------------------------------------------
static void testDeterminism() {
  std::printf("determinism\n");
  PairIndex pix(16);
  auto once = [&]() {
    Rng rng(999);
    Graph s = Graph::randomConnected(pix, 45, rng);
    Oracle o;
    GA::Config cfg; cfg.popSize = 16; cfg.generations = 12;
    return GA(cfg).run(s, o, rng);
  };
  Result a = once(), b = once();
  expectTrue(a.lam2 == b.lam2 && a.edges == b.edges && a.oracleCalls == b.oracleCalls,
             "same seed reproduces bit-for-bit");
}


// ---- ported CDC 2026 companion baselines (MDMD, KOPT) -------------
static void testPortedConstructorsFeasible() {
  std::printf("ported baselines: feasibility, determinism, budget accounting\n");
  for (int n : {10, 16, 24}) {
    PairIndex pix(n);
    for (int m : {n + 3, 2 * n}) {
      if (m > pix.numPairs()) continue;
      Rng rng(7 * n + m);
      Graph start = Graph::randomConnected(pix, m, rng);

      {  // MDMD: zero eigensolves during construction, exactly one at the end.
        Oracle o;
        Rng r1(1);
        Result res = MDMD().run(start, o, r1);
        expectTrue((int)res.edges.size() == m, "mdmd returns exactly m edges");
        expectTrue(o.calls() == 1, "mdmd spends exactly one oracle call");
        expectNear(res.lam2, lam2Of(fromEdges(pix, res.edges)), 1e-9,
                   "mdmd reported lambda_2 matches the returned graph");
        Oracle o2;
        Rng r2(2);
        Result res2 = MDMD().run(start, o2, r2);
        expectTrue(res.edges == res2.edges, "mdmd is deterministic");
      }
      {  // KOPT: starts at G0 and never returns anything worse.
        Oracle o;
        Rng r1(1);
        const double before = lam2Of(start);
        Result res = KOPT().run(start, o, r1);
        expectTrue((int)res.edges.size() == m, "kopt returns exactly m edges");
        expectTrue(res.lam2 >= before - 1e-12, "kopt never decreases lambda_2");
        expectTrue(fromEdges(pix, res.edges).isConnected(), "kopt output connected");
        expectNear(res.lam2, lam2Of(fromEdges(pix, res.edges)), 1e-9,
                   "kopt reported lambda_2 matches the returned graph");
        Oracle o2;
        Rng r2(9);
        Result res2 = KOPT().run(start, o2, r2);
        expectTrue(res.edges == res2.edges,
                   "kopt (k=1) is deterministic regardless of rng seed");
      }
    }
  }
}


// ---- effective resistance: Foster's theorem and Sherman-Morrison drift -----
// Foster: sum over the m EDGES of a connected graph of R_eff(e) = n - 1,
// exactly, for every connected graph. It pins the whole inverse with one
// scalar, so it catches both a wrong formula and accumulated rank-one drift.
static void testGroundedInverse() {
  std::printf("grounded inverse: Foster's theorem, incremental vs exact\n");
  for (int n : {6, 12, 20}) {
    PairIndex pix(n);
    for (int m : {n, 2 * n, 3 * n}) {
      if (m > pix.numPairs()) continue;
      Rng rng(31 * n + m);
      Graph g = Graph::randomConnected(pix, m, rng);

      detail::GroundedInverse inv(pix, g.edges());
      double foster = 0.0;
      for (int k = 0; k < pix.numPairs(); ++k)
        if (g.edges().test(k)) foster += inv.effectiveResistance(pix.i(k), pix.j(k));
      expectNear(foster, n - 1, 1e-7, "Foster: sum of edge resistances = n-1");

      // Build the SAME graph incrementally from its spanning tree and compare
      // against a from-scratch inverse. Any Sherman-Morrison drift shows here.
      Bitset e = detail::bfsSpanningTree(g);
      detail::GroundedInverse incr(pix, e);
      for (int k = 0; k < pix.numPairs(); ++k) {
        if (!g.edges().test(k) || e.test(k)) continue;
        e.set(k);
        incr.addEdge(pix.i(k), pix.j(k), e);
      }
      double worst = 0.0;
      for (int k = 0; k < pix.numPairs(); ++k)
        worst = std::max(worst, std::abs(incr.effectiveResistance(pix.i(k), pix.j(k)) -
                                         inv.effectiveResistance(pix.i(k), pix.j(k))));
      expectTrue(worst < 1e-8, "incremental inverse matches exact inverse");
    }
  }
}

// ---- ErGreedy and RANDOM -----------------------------------------
static void testErGreedyAndRandom() {
  std::printf("er / random: feasibility, budget, determinism\n");
  for (int n : {10, 16, 24}) {
    PairIndex pix(n);
    for (int m : {n + 3, 2 * n}) {
      if (m > pix.numPairs()) continue;
      Rng rng(13 * n + m);
      Graph start = Graph::randomConnected(pix, m, rng);

      {  // RANDOM floor: same seed reproduces, different seeds may differ.
        Oracle o;
        Rng r1(5);
        Result res = RANDOM().run(start, o, r1);
        expectTrue((int)res.edges.size() == m, "random returns exactly m edges");
        expectTrue(o.calls() == 1, "random spends exactly one oracle call");
        expectTrue(fromEdges(pix, res.edges).isConnected(), "random output connected");
        expectNear(res.lam2, lam2Of(fromEdges(pix, res.edges)), 1e-9,
                   "random reported lambda_2 matches the returned graph");
        Oracle o2;
        Rng r2(5);
        Result res2 = RANDOM().run(start, o2, r2);
        expectTrue(res.edges == res2.edges, "random reproduces from the same seed");
      }
    }
  }
}



// ---- ER: the walk contract -------------------------------------------
// ER is now a free score-greedy walk: moves cost no calls, walks end at the
// first revisited graph, and only bound-gated evaluations spend budget. The
// admissibility contract is unchanged: spend at most B, monotone in B,
// deterministic per seed, and return an evaluated feasible graph.
static void testErBudget() {
  std::printf("er: budget respected, monotone in B, cycle-terminated walk\n");
  for (int n : {10, 16, 24}) {
    PairIndex pix(n);
    for (int m : {n + 3, 2 * n}) {
      if (m > pix.numPairs()) continue;
      Rng rng(23 * n + m);
      Graph start = Graph::randomConnected(pix, m, rng);
      const double lam0 = lam2Of(start);

      double prev = -1.0;
      for (long B : {5L, 25L, 100L}) {
        Oracle o(B);
        Rng r(4);
        Result res = ER().run(start, o, r);
        expectTrue(o.calls() <= B, "er never exceeds its budget");
        expectTrue((int)res.edges.size() == m, "er returns exactly m edges");
        expectTrue(fromEdges(pix, res.edges).isConnected(), "er output connected");
        expectTrue(res.lam2 >= lam0 - 1e-12, "er never returns worse than G0");
        expectNear(res.lam2, lam2Of(fromEdges(pix, res.edges)), 1e-9,
                   "er reported lambda_2 matches the returned graph");
        expectTrue(res.lam2 >= prev - 1e-12, "er is monotone in the budget");
        prev = res.lam2;
      }

      {  // it must actually USE the budget, not halt at one call like the
         // constructor form did -- that is exactly what made it inadmissible.
        Oracle o(200);
        Rng r(4);
        ER().run(start, o, r);
        expectTrue(o.calls() > 1, "er consumes budget rather than a single call");
      }
      {  // same seed, same answer
        Oracle o1(50), o2(50);
        Rng a(11), b(11);
        expectTrue(ER().run(start, o1, a).edges == ER().run(start, o2, b).edges,
                   "er reproduces from the same seed");
      }
    }
  }
}

// FV must obey the same contract now that its restart is gated on the
// degeneracy certificate rather than on reaching any leaf.
static void testFvBudget() {
  std::printf("fv: budget respected, monotone in B\n");
  for (int n : {12, 20}) {
    PairIndex pix(n);
    const int m = 2 * n;
    if (m > pix.numPairs()) continue;
    Rng rng(5 * n);
    Graph start = Graph::randomConnected(pix, m, rng);
    double prev = -1.0;
    for (long B : {10L, 50L, 200L}) {
      Oracle o(B);
      Rng r(3);
      Result res = FV().run(start, o, r);
      expectTrue(o.calls() <= B, "fv never exceeds its budget");
      expectTrue((int)res.edges.size() == m, "fv returns exactly m edges");
      expectTrue(res.lam2 >= prev - 1e-12, "fv is monotone in the budget");
      prev = res.lam2;
    }
  }
}


// The inertia-bisection evaluator must reproduce the dense eigensolver's
// lambda_2 for arbitrary swaps -- it is OURS's zero-cost neighbourhood.
static void testLam2AfterSwap() {
  std::printf("exact swap evaluation via inertia bisection\n");
  for (int n : {8, 16, 24}) {
    PairIndex pix(n);
    Rng rng(77 * n);
    for (int rep = 0; rep < 8; ++rep) {
      Graph g = Graph::randomConnected(pix, 2 * n, rng);
      std::vector<double> vecs = g.laplacian(), vals;
      eig::symmetric(vecs, n, vals, true);
      int checked = 0;
      for (int k = 0; k < pix.numPairs() && checked < 12; ++k) {
        if (g.has(k)) continue;
        for (int e = 0; e < pix.numPairs() && checked < 12; ++e) {
          if (!g.edges().test(e)) continue;
          const Swap s{k, e};
          if (!g.swapKeepsConnected(s)) continue;
          Graph h = g;
          h.applySwap(s);
          std::vector<double> alpha(n), beta(n);
          for (int i = 0; i < n; ++i) {
            alpha[i] = vecs[(size_t)pix.i(k) * n + i] - vecs[(size_t)pix.j(k) * n + i];
            beta[i] = vecs[(size_t)pix.i(e) * n + i] - vecs[(size_t)pix.j(e) * n + i];
          }
          expectNear(eig::lam2AfterSwap(vals, alpha, beta), lam2Of(h), 1e-8,
                     "lam2AfterSwap matches dense recompute");
          const double swapTruth = lam2Of(h);
          for (double threshold : {
                   vals[0] + (vals[1] - vals[0]) / 2.0,
                   vals[1] + (vals[2] - vals[1]) / 2.0}) {
            if (std::binary_search(vals.begin(), vals.end(), threshold)) continue;
            const eig::ShiftedSwapInertia shared(vals, vecs, threshold);
            const int sharedCount = shared.countBelow(
                pix.i(k), pix.j(k), pix.i(e), pix.j(e));
            const auto addProjection =
                shared.projectPair(pix.i(k), pix.j(k));
            const auto removeProjection =
                shared.projectPair(pix.i(e), pix.j(e));
            expectTrue(shared.countBelow(addProjection, removeProjection) ==
                           sharedCount,
                       "precomputed pair projection preserves inertia count");
            expectTrue(sharedCount ==
                           eig::swapCountBelow(vals, alpha, beta, threshold),
                       "shared shifted resolvent matches spectral-sum inertia");
            expectTrue(shared.lam2AtLeast(
                           pix.i(k), pix.j(k), pix.i(e), pix.j(e)) ==
                           (swapTruth >= threshold),
                       "shared shifted resolvent matches dense swap decision");
          }
          // Composite rank-4 decision probe: 2-swap = this swap plus one
          // more; check the inertia decision against the dense recompute.
          for (int k2 = k + 1; k2 < pix.numPairs(); ++k2) {
            if (h.has(k2)) continue;
            for (int e2 = 0; e2 < pix.numPairs(); ++e2) {
              if (!h.edges().test(e2) || e2 == k) continue;
              Graph h2 = h;
              if (!h2.swapKeepsConnected(Swap{k2, e2})) continue;
              h2.applySwap(Swap{k2, e2});
              std::vector<double> co(4 * (size_t)n);
              std::vector<int> sg = {+1, +1, -1, -1};
              int pr[4][2] = {{pix.i(k), pix.j(k)}, {pix.i(k2), pix.j(k2)},
                              {pix.i(e), pix.j(e)}, {pix.i(e2), pix.j(e2)}};
              for (int q = 0; q < 4; ++q)
                for (int i = 0; i < n; ++i)
                  co[(size_t)q * n + i] =
                      vecs[(size_t)pr[q][0] * n + i] - vecs[(size_t)pr[q][1] * n + i];
              const double truth = lam2Of(h2);
              for (double thr : {truth - 1e-6, truth + 1e-6})
                expectTrue(eig::lam2AfterEditAtLeast(vals, co, sg, thr) == (truth >= thr),
                           "rank-4 probe decision matches dense recompute");
              break;
            }
            break;
          }
          ++checked;
          break;  // next add pair
        }
      }
    }
  }
}

// The oracle's lambda_3 must be the eigenvalue above lambda_2, since the
// interlacing escape test in OURS rides on it.
static void testOracleLam3() {
  std::printf("oracle: lambda_3 from the same call\n");
  const int n = 9;
  PairIndex pix(n);
  std::vector<std::pair<int, int>> K, S;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) K.push_back({i, j});
  for (int j = 1; j < n; ++j) S.push_back({0, j});
  {  // K_n: spectrum {0, n^(n-1)}, so lambda_3 = n
    Oracle o;
    expectNear(o.lam3(fromEdges(pix, K)), n, 1e-9, "lambda3(K_n) = n");
  }
  {  // star: spectrum {0, 1^(n-2), n}, so lambda_3 = 1
    Oracle o;
    expectNear(o.lam3(fromEdges(pix, S)), 1.0, 1e-9, "lambda3(star_n) = 1");
  }
  {  // lam3 never re-charges and always dominates lam2
    Oracle o(1);
    Rng rng(7);
    const Graph g = Graph::randomConnected(pix, 2 * n, rng);
    const double l2 = o.lam2(g);
    const double l3 = o.lam3(g);  // cached: must not throw at budget 1
    expectTrue(l3 >= l2 - 1e-12, "lambda3 >= lambda2");
    expectTrue(o.calls() == 1, "lam3 costs nothing beyond the paid call");
  }
}

// OURS must obey the same contract as FV and ER: budget respected, feasible
// output, monotone in B, deterministic, never below its start.
static void testOursBudget() {
  std::printf("ours: budget respected, monotone in B, feasible, deterministic\n");
  for (int n : {10, 16, 24}) {
    PairIndex pix(n);
    for (int m : {n + 3, 2 * n}) {
      if (m > pix.numPairs()) continue;
      Rng rng(31 * n + m);
      Graph start = Graph::randomConnected(pix, m, rng);
      const double lam0 = lam2Of(start);

      double prev = -1.0;
      for (long B : {5L, 25L, 100L, 400L}) {
        Oracle o(B);
        Rng r(6);
        Result res = OURS().run(start, o, r);
        expectTrue(o.calls() <= B, "ours never exceeds its budget");
        expectTrue(o.calls() == o.realSolves(),
                   "every ours budget unit is one physical eigensolve");
        expectTrue(res.steps <=
                       B * std::min(m, pix.numPairs() - m),
                   "every paid solve has the structural simple-path move bound");
        expectTrue((int)res.edges.size() == m, "ours returns exactly m edges");
        expectTrue(fromEdges(pix, res.edges).isConnected(), "ours output connected");
        expectTrue(res.lam2 >= lam0 - 1e-12, "ours never returns worse than G0");
        expectNear(res.lam2, lam2Of(fromEdges(pix, res.edges)), 1e-9,
                   "ours reported lambda_2 matches the returned graph");
        expectTrue(res.lam2 <= res.momentBound + 1e-9,
                   "ours respects the certified moment bound");
        expectTrue(res.lam2 >= prev - 1e-12, "ours is monotone in the budget");
        prev = res.lam2;
      }

      {  // it must consume budget rather than halting at one call
        Oracle o(200);
        Rng r(6);
        OURS().run(start, o, r);
        expectTrue(o.calls() > 1, "ours consumes budget rather than a single call");
      }
      {  // same seed, same answer
        Oracle o1(50), o2(50);
        Rng a(11), b(11);
        expectTrue(OURS().run(start, o1, a).edges == OURS().run(start, o2, b).edges,
                   "ours reproduces from the same seed");
      }
      {  // the exact climb must convert essentially every call into a move:
         // a paid call is only ever spent verifying a secular-certified
         // argmax or sampling a restart, never on speculative proposals.
        OURS::Config off;
        off.attack = false;
        Oracle o(60);
        Rng r(9);
        const Result res = OURS(off).run(start, o, r);
        expectTrue(res.lam2 >= lam2Of(start) - 1e-12,
                   "ours(attack=false) climbs from the start");
      }
    }
  }

  {  // Controlled ablations preserve feasibility, accounting, and reporting.
    const int n = 10, m = 18;
    PairIndex pix(n);
    Rng startRng(710);
    const Graph start = Graph::randomConnected(pix, m, startRng);
    std::vector<OURS::Config> variants;
    OURS::Config separable;
    separable.includeCoupling = false;
    variants.push_back(separable);
    OURS::Config noInertia;
    noInertia.attack = false;
    noInertia.inertiaScreen = false;
    variants.push_back(noInertia);
    OURS::Config narrow;
    narrow.proposalWidth = (n + 1) / 2;
    variants.push_back(narrow);
    OURS::Config wide;
    wide.proposalWidth = 2 * n;
    variants.push_back(wide);
    for (size_t i = 0; i < variants.size(); ++i) {
      Oracle oracle(60);
      Rng runRng(81);
      const Result result = OURS(variants[i]).run(start, oracle, runRng);
      const Graph returned = fromEdges(pix, result.edges);
      expectTrue(oracle.calls() <= 60,
                 "ours ablation respects the eigensolve budget");
      expectTrue(returned.isConnected() && returned.m() == m,
                 "ours ablation returns a feasible graph");
      expectNear(result.lam2, lam2Of(returned), 1e-9,
                 "ours ablation reports its returned graph");
    }
  }
}

// The proposal oracle examines the rank-matched union of complete rows and
// columns and retains its exact bidirectional best-response closure. The work
// bound is measured by every executed body of the production union loops, not
// inferred from output size. There is no enclosing Cartesian-product scan.
static void testOursCoordinateClosure() {
  std::printf("ours: bounded rank-matched coupled closure\n");
  for (int n : {8, 12, 20}) {
    PairIndex pix(n);
    for (int m : {n - 1, 2 * n, pix.numPairs() - 1}) {
      if (m < n - 1 || m >= pix.numPairs()) continue;
      Rng rng(static_cast<uint64_t>(n * 1000 + m));
      const Graph g = Graph::randomConnected(pix, m, rng);
      const std::vector<Swap> proposals = OURS::coupledProposals(g);
      expectTrue(!proposals.empty(), "coupled proposal closure is nonempty");
      expectTrue((int)proposals.size() <= pix.numPairs(),
                 "coupled proposal closure has its structural A+R bound");

      detail::GroundedInverse inverse(pix, g.edges(), 0, 0, 0.0);
      std::vector<double> resistance(pix.numPairs(), 0.0);
      std::vector<int> additions, removals;
      for (int k = 0; k < pix.numPairs(); ++k) {
        resistance[k] =
            inverse.effectiveResistance(pix.i(k), pix.j(k));
        (g.has(k) ? removals : additions).push_back(k);
      }
      std::sort(additions.begin(), additions.end(), [&](int left, int right) {
        if (resistance[left] != resistance[right])
          return resistance[left] > resistance[right];
        return left < right;
      });
      std::sort(removals.begin(), removals.end(), [&](int left, int right) {
        if (resistance[left] != resistance[right])
          return resistance[left] < resistance[right];
        return left < right;
      });
      const size_t rank = static_cast<size_t>(n - 1);
      const size_t addRows = std::min(rank, additions.size());
      const size_t removalColumns = std::min(rank, removals.size());
      const size_t expectedExamined = addRows * removals.size() +
          additions.size() * removalColumns - addRows * removalColumns;
      const size_t examined = OURS::coupledProposalPairsExamined(g);
      expectTrue(examined == expectedExamined,
                 "counter equals the rank-row/column union cardinality");
      expectTrue(examined <= rank * static_cast<size_t>(pix.numPairs()),
                 "proposal examination is bounded by (n-1) binom(n,2)");
      for (int radius = 1;
           radius <= std::min(n - 1,
                              std::min(m, pix.numPairs() - m));
           ++radius) {
        const size_t continuation =
            OURS::cubicContinuationPairBound(g, radius);
        const size_t expected = static_cast<size_t>(radius - 1) *
            static_cast<size_t>(pix.numPairs() - radius);
        expectTrue(continuation == expected,
                   "composite pair certificate equals the level sum");
        expectTrue(continuation <=
                       static_cast<size_t>(n - 2) * pix.numPairs(),
                   "composite coordinate work is structurally O(n^3)");
      }
      std::vector<int> addRank(pix.numPairs(), -1), removalRank(pix.numPairs(), -1);
      for (size_t i = 0; i < additions.size(); ++i) addRank[additions[i]] = i;
      for (size_t i = 0; i < removals.size(); ++i) removalRank[removals[i]] = i;

      auto scoreOf = [&](int add, int remove) {
        const double addR =
            inverse.effectiveResistance(pix.i(add), pix.j(add));
        const double removeR =
            inverse.effectiveResistance(pix.i(remove), pix.j(remove));
        const double transfer = inverse.transfer(
            pix.i(add), pix.j(add), pix.i(remove), pix.j(remove));
        return addR - removeR - addR * removeR + transfer * transfer;
      };
      double unionBest = -std::numeric_limits<double>::infinity();
      for (size_t ai = 0; ai < additions.size(); ++ai)
        for (size_t ri = 0; ri < removals.size(); ++ri)
          if (ai < addRows || ri < removalColumns)
            unionBest = std::max(
                unionBest, scoreOf(additions[ai], removals[ri]));

      std::set<std::pair<int, int>> unique;
      double previousScore = std::numeric_limits<double>::infinity();
      for (Swap swap : proposals) {
        expectTrue(!g.has(swap.add) && g.has(swap.remove),
                   "every coupled proposal is a completed add/remove swap");
        expectTrue(unique.insert({swap.add, swap.remove}).second,
                   "coupled closure contains no duplicate pair");
        expectTrue(addRank[swap.add] >= 0 && removalRank[swap.remove] >= 0 &&
                       (static_cast<size_t>(addRank[swap.add]) < addRows ||
                        static_cast<size_t>(removalRank[swap.remove]) <
                            removalColumns),
                   "every proposal belongs to the rank-matched union");
        const double score = scoreOf(swap.add, swap.remove);
        expectTrue(score <= previousScore ||
                       std::nextafter(score,
                                      std::numeric_limits<double>::infinity()) >=
                           previousScore,
                   "coupled proposals are ordered by completed-swap score");
        previousScore = score;
      }
      expectTrue(proposals.empty() || scoreOf(proposals.front().add,
                                              proposals.front().remove) ==
                                           unionBest,
                 "closure contains the rank-matched union maximizer");
    }
  }
}

// ---- the trace must track FEASIBLE graphs only -----------------------------
// Regression: an off-manifold evaluation (more than m edges) has a larger
// lambda_2 by monotonicity. If it were allowed to set the running best, every
// quality-vs-budget curve for the GA would be inflated.
static void testTraceFeasibleOnly() {
  std::printf("oracle trace: off-manifold calls cannot set the best\n");
  const int n = 12, m = 2 * n;
  PairIndex pix(n);
  Rng rng(4242);
  Graph g = Graph::randomConnected(pix, m, rng);

  Oracle o;
  const double lamFeasible = o.lam2(g);            // first call fixes refM_ = m
  expectNear(o.best(), lamFeasible, 1e-12, "best starts at the feasible value");

  // Build a denser graph on the same vertex set by adding edges directly.
  Bitset denser = g.edges();
  int added = 0;
  for (int k = 0; k < pix.numPairs() && added < 5; ++k)
    if (!denser.test(k)) { denser.set(k); ++added; }
  Graph fat(pix, std::move(denser));
  const double lamFat = o.lam2(fat);

  expectTrue(lamFat > lamFeasible, "denser graph really does have larger lambda_2");
  expectTrue(o.callsOffManifold() == 1, "the denser call is counted off-manifold");
  expectNear(o.best(), lamFeasible, 1e-12,
             "off-manifold lambda_2 does NOT become the running best");
  for (auto [calls, best] : o.trace())
    expectTrue(best <= lamFeasible + 1e-12, "no trace point exceeds the feasible best");
}


// ---- parallel independent restarts ---------------------------------------
// FV and ER restart from a fresh random graph, so their climbs are i.i.d. and
// K of them may run concurrently on B/K each. The property that must hold is
// that the BUDGET is conserved: K workers spend no more eigensolves between
// them than one worker given B. Wall-clock falls; the reported cost does not.
static void testParallelRestarts() {
  std::printf("parallel restarts: budget conserved, deterministic, merged trace sane\n");
  for (int n : {12, 20}) {
    PairIndex pix(n);
    const int m = 2 * n;
    if (m > pix.numPairs()) continue;
    Rng seedRng(77 * n);
    Graph start = Graph::randomConnected(pix, m, seedRng);
    const long B = 400;

    for (int K : {1, 2, 4, 8}) {
      ParallelResult pr = runParallelRestarts(FV(), start, B, K, 1234);
      expectTrue(pr.totalCalls <= B, "K workers never exceed the shared budget");
      expectTrue(pr.totalRealSolves == pr.totalCalls,
                 "parallel FV reports every physical eigensolve");
      expectTrue((int)pr.best.edges.size() == m, "parallel result has exactly m edges");
      expectTrue(fromEdges(pix, pr.best.edges).isConnected(), "parallel result connected");
      expectNear(pr.best.lam2, lam2Of(fromEdges(pix, pr.best.edges)), 1e-9,
                 "reported lambda_2 matches the returned graph");
      // the merged curve is a non-decreasing step function in TOTAL calls
      long prevC = -1; double prevV = -1.0;
      for (auto [c, v] : pr.trace) {
        expectTrue(c > prevC, "merged trace strictly advances in calls");
        expectTrue(v > prevV, "merged trace strictly improves");
        prevC = c; prevV = v;
      }
      if (!pr.trace.empty()) {
        expectNear(pr.trace.back().second, pr.best.lam2, 1e-9,
                   "merged trace ends at the reported best");
        // Round-robin interleaving: the first call of worker 0 IS the first
        // call overall, so the curve must begin at B = 1 regardless of K.
        // Mapping c -> c*K instead would start it at K and make runs with
        // different worker counts incomparable at small B.
        expectTrue(pr.trace.front().first == 1,
                   "merged trace starts at B = 1 for every worker count");
      }
      // same (seed, K) reproduces bit-for-bit despite the threading
      ParallelResult again = runParallelRestarts(FV(), start, B, K, 1234);
      expectTrue(again.best.edges == pr.best.edges && again.totalCalls == pr.totalCalls,
                 "parallel restarts are deterministic for fixed (seed, K)");
    }
    // ER obeys the same contract
    ParallelResult pe = runParallelRestarts(ER(), start, B, 4, 99);
    expectTrue(pe.totalCalls <= B, "ER: K workers never exceed the shared budget");
    expectTrue(pe.totalRealSolves == pe.totalCalls,
               "parallel ER reports every physical eigensolve");
    expectTrue((int)pe.best.edges.size() == m, "ER: parallel result has exactly m edges");
  }
}


// ---- k-opt must be budget-matchable --------------------------------------
// The reference implementation caps rounds at 20, which pins the method at
// ~8.4e3 evaluations no matter how large B is. Under a matched budget that is
// disqualifying: the method would be reported at B while structurally unable
// to spend it. The paper itself iterates "until no further improvements can be
// made", so rounds are budget-bound and a converged search restarts.
static void testKOptUsesBudget() {
  std::printf("k-opt: consumes the budget, monotone in B, restarts on convergence\n");
  const int n = 24, m = 3 * n;
  PairIndex pix(n);
  Rng seedRng(31 * n);
  Graph start = Graph::randomConnected(pix, m, seedRng);
  const double lam0 = lam2Of(start);

  double prev = -1.0;
  long prevCalls = 0;
  for (long B : {2000L, 20000L}) {
    Oracle o(B);
    Rng r(5);
    Result res = KOPT().run(start, o, r);
    expectTrue(o.calls() <= B, "k-opt never exceeds its budget");
    expectTrue((int)res.edges.size() == m, "k-opt returns exactly m edges");
    expectTrue(fromEdges(pix, res.edges).isConnected(), "k-opt output connected");
    expectTrue(res.lam2 >= lam0 - 1e-12, "k-opt never returns worse than G0");
    expectNear(res.lam2, lam2Of(fromEdges(pix, res.edges)), 1e-9,
               "k-opt reported lambda_2 matches the returned graph");
    expectTrue(res.lam2 >= prev - 1e-12, "k-opt is monotone in the budget");
    prev = res.lam2;
    prevCalls = o.calls();
  }
  // the point of the change: a large budget is actually spent, not capped at
  // the ~8.4e3 the fixed round limit allowed
  expectTrue(prevCalls > 9000, "k-opt spends past the old fixed round cap");
}


// ---- concatenated tapes must match a sequential run ----------------------
// Restarts are i.i.d., so K workers' tapes concatenated are the same sequence
// a single worker would have produced. The test that matters is therefore not
// "does it run" but "is a K>1 curve comparable to a K=1 curve": the number of
// improvements recorded below a small budget must not collapse by a factor of
// K, which is exactly what the old wall-clock interleaving did.
static void testTapesComparableAcrossK() {
  std::printf("parallel tapes: K>1 curve comparable with K=1 at small B\n");
  const int n = 24, m = 3 * n;
  PairIndex pix(n);
  Rng seedRng(91 * n);
  Graph start = Graph::randomConnected(pix, m, seedRng);
  const long B = 4000;

  auto below = [](const ParallelResult& r, long cap) {
    int k = 0;
    for (auto [c, v] : r.trace) { (void)v; if (c <= cap) ++k; }
    return k;
  };
  const ParallelResult one = runParallelRestarts(ER(), start, B, 1, 2024);
  for (int K : {2, 4, 8}) {
    const ParallelResult many = runParallelRestarts(ER(), start, B, K, 2024);
    expectTrue(many.totalCalls <= B, "K workers stay within the shared budget");
    expectTrue(many.trace.front().first == 1, "tape starts at B = 1");
    // The old bug made this ratio ~1/K. Allow generous slack for search noise
    // but catch a systematic collapse.
    const int a = below(one, 200), b = below(many, 200);
    expectTrue(b * 3 >= a,
               "improvements below B=200 do not collapse by a factor of K");
    expectTrue(many.trace.back().first <= B, "tape never exceeds the budget");
  }
}


// ---- rewiring updates of the grounded inverse ----------------------------
// Regression for a state-mismatch bug in ER: removeEdge/addEdge take the
// bitset they may REBUILD from, and ER handed both calls the final post-swap
// state. A periodic recompute inside removeEdge then rebuilt from a state
// already containing the added edge, and the following addEdge counted it
// twice. resetEvery = 1 forces that recompute on every operation, so this
// test walks a long swap sequence under the correct discipline (intermediate
// state for the removal, final state for the addition) and checks the
// resistances against a from-scratch inverse at every step.
static void testGroundedInverseSwapDiscipline() {
  std::printf("grounded inverse: swap updates stay exact under forced recompute\n");
  const int n = 16, m = 40;
  PairIndex pix(n);
  Rng rng(777);
  Graph g = Graph::randomConnected(pix, m, rng);
  for (int resetEvery : {1, 2, 32}) {
    Graph cur = g;
    detail::GroundedInverse inv(pix, cur.edges(), 0, resetEvery);
    int applied = 0;
    for (int t = 0; t < 400 && applied < 60; ++t) {
      const int add = (int)rng.below((uint32_t)pix.numPairs());
      const int del = (int)rng.below((uint32_t)pix.numPairs());
      if (cur.has(add) || !cur.has(del)) continue;
      Swap s{add, del};
      if (!cur.swapKeepsConnected(s)) continue;
      Graph trial = cur;
      trial.applySwap(s);
      Bitset mid = cur.edges();
      mid.clear(s.remove);
      inv.removeEdge(pix.i(s.remove), pix.j(s.remove), mid);
      inv.addEdge(pix.i(s.add), pix.j(s.add), trial.edges());
      cur = std::move(trial);
      ++applied;

      detail::GroundedInverse fresh(pix, cur.edges());
      double worst = 0.0;
      for (int k = 0; k < pix.numPairs(); ++k)
        worst = std::max(worst,
                         std::abs(inv.effectiveResistance(pix.i(k), pix.j(k)) -
                                  fresh.effectiveResistance(pix.i(k), pix.j(k))));
      expectTrue(worst < 1e-6, "incremental inverse tracks the graph through swaps");
    }
    expectTrue(applied >= 40, "the walk actually applied a long swap sequence");
  }

  // OURS uses the strict forms: no periodic reset, no ridge, and no hidden
  // fallback factorization. Exact completed-swap connectivity makes both
  // denominators positive; the fallible O(n^2) updates must therefore carry a
  // representative path while agreeing with a fresh inverse.
  {
    Graph cur = g;
    detail::GroundedInverse inv(pix, cur.edges(), 0, 0, 0.0);
    int applied = 0;
    for (int t = 0; t < 500 && applied < 60; ++t) {
      const int add = static_cast<int>(
          rng.below(static_cast<uint32_t>(pix.numPairs())));
      const int del = static_cast<int>(
          rng.below(static_cast<uint32_t>(pix.numPairs())));
      if (cur.has(add) || !cur.has(del)) continue;
      const Swap swap{add, del};
      if (!cur.swapKeepsConnected(swap)) continue;
      Graph trial = cur;
      trial.applySwap(swap);
      const bool addOk = inv.tryAddEdge(pix.i(add), pix.j(add));
      const bool removeOk =
          addOk && inv.tryRemoveEdge(pix.i(del), pix.j(del));
      expectTrue(addOk && removeOk,
                 "strict inverse updates accept a connected completed swap");
      if (!addOk || !removeOk) break;
      cur = std::move(trial);
      ++applied;

      detail::GroundedInverse fresh(pix, cur.edges(), 0, 0, 0.0);
      double worst = 0.0;
      for (int k = 0; k < pix.numPairs(); ++k)
        worst = std::max(
            worst,
            std::abs(inv.effectiveResistance(pix.i(k), pix.j(k)) -
                     fresh.effectiveResistance(pix.i(k), pix.j(k))));
      expectTrue(worst < 1e-6,
                 "strict O(n^2) updates match a fresh inverse");
    }
    expectTrue(applied >= 40,
               "strict inverse test exercises a long completed-swap path");
  }
}


int main() {
  testEigenClosedForms();
  testValuesOnlyPathAgrees();
  testMultiplicity();
  testInvariants();
  testDegreeCaches();
  testMomentBound();
  testBridges();
  testSwapConnectivity();
  testNonBridgeRemovalKeepsConnected();
  testDegeneracyImpliesLocalOptimum();
  testMomentBounds();
  testOracle();
  testFvClimb();
  testCrossoverIsRewire();
  testGa();
  testCrossoverPruneEqualsReverseDelete();
  testGaBeatsFvClimbOnAverage();
  testSelectionPressure();
  testDeterminism();
  testPortedConstructorsFeasible();
  testGroundedInverse();
  testGroundedInverseSwapDiscipline();
  testErGreedyAndRandom();
  testErBudget();
  testFvBudget();
  testOracleLam3();
  testLam2AfterSwap();
  testOursBudget();
  testOursCoordinateClosure();
  testTraceFeasibleOnly();
  testParallelRestarts();
  testKOptUsesBudget();
  testTapesComparableAcrossK();
  std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}

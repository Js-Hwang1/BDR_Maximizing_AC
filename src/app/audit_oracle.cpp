// Accuracy audit of the two numerical layers the method rests on.
//
//   LAYER 1, the oracle: tred2 + implicit-QL. Checked against closed-form
//   spectra (K_n, C_n, P_n, star, K_{a,b}) and by eigenpair residual.
//
//   LAYER 2, the probes: the Haynsworth inertia decisions that replace
//   eigensolves between evaluations. Checked against a DENSE eigensolve of
//   every swapped Laplacian -- both the value returned by bisection and, what
//   actually matters to the search, the DECISION at the acceptance threshold.
//
// The decision errors are asymmetric in cost:
//   false NEGATIVE  probe says "no improvement", truth says yes -- an improving
//                   move is silently lost, and the search never learns of it;
//   false POSITIVE  probe says yes, truth says no -- costs one wasted
//                   evaluation, which the rollback path already handles.
// A false negative rate above noise means the method is leaving value on the
// table for numerical, not algorithmic, reasons.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "Eigen.hpp"
#include "Bitset.hpp"
#include "Graph.hpp"
#include "PairIndex.hpp"
#include "Rng.hpp"

using namespace algconn;

namespace {

void laplacian(const Graph& g, int n, std::vector<double>& L) {
  L.assign(static_cast<size_t>(n) * n, 0.0);
  const PairIndex& pix = g.pix();
  for (int k = 0; k < pix.numPairs(); ++k) {
    if (!g.has(k)) continue;
    const int u = pix.i(k), v = pix.j(k);
    L[static_cast<size_t>(u) * n + v] -= 1.0;
    L[static_cast<size_t>(v) * n + u] -= 1.0;
    L[static_cast<size_t>(u) * n + u] += 1.0;
    L[static_cast<size_t>(v) * n + v] += 1.0;
  }
}

// Dense spectrum, and (optionally) the eigenvectors, of a graph Laplacian.
void spectrum(const Graph& g, int n, std::vector<double>& vals,
              std::vector<double>* vecs) {
  std::vector<double> L;
  laplacian(g, n, L);
  eig::symmetric(L, n, vals, vecs != nullptr);
  if (vecs) *vecs = L;
}

double maxResidual(const Graph& g, int n) {
  std::vector<double> L, vals, vecs;
  laplacian(g, n, L);
  spectrum(g, n, vals, &vecs);
  double worst = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      double s = 0.0;
      for (int k = 0; k < n; ++k)
        s += L[static_cast<size_t>(i) * n + k] * vecs[static_cast<size_t>(k) * n + j];
      worst = std::max(worst, std::fabs(s - vals[j] * vecs[static_cast<size_t>(i) * n + j]));
    }
  }
  return worst;
}

}  // namespace

int main(int argc, char** argv) {
  int n = 24, trials = 40;
  uint64_t seed = 4242;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--n") n = std::stoi(argv[++i]);
    else if (a == "--trials") trials = std::stoi(argv[++i]);
    else if (a == "--seed") seed = std::stoull(argv[++i]);
  }
  PairIndex pix(n);

  // ---- LAYER 1: closed-form spectra ------------------------------------
  std::printf("== oracle vs closed form (n=%d) ==\n", n);
  {
    auto build = [&](const std::vector<int>& pairs) {
      Bitset b(pix.numPairs());
      for (int k : pairs) b.set(k);
      return Graph(pix, b);
    };
    std::vector<int> all;
    for (int k = 0; k < pix.numPairs(); ++k) all.push_back(k);
    const Graph kn = build(all);
    std::vector<double> v;
    spectrum(kn, n, v, nullptr);
    std::printf("  K_n     lambda_2 = %.16f  exact %d      err %.3e\n", v[1], n,
                std::fabs(v[1] - n));

    std::vector<int> cp;
    for (int i = 0; i < n; ++i) cp.push_back(pix.index(i, (i + 1) % n));
    spectrum(build(cp), n, v, nullptr);
    const double ce = 2.0 * (1.0 - std::cos(2.0 * M_PI / n));
    std::printf("  C_n     lambda_2 = %.16f  exact %.16f  err %.3e\n", v[1], ce,
                std::fabs(v[1] - ce));

    std::vector<int> pp;
    for (int i = 0; i + 1 < n; ++i) pp.push_back(pix.index(i, i + 1));
    spectrum(build(pp), n, v, nullptr);
    const double pe = 2.0 * (1.0 - std::cos(M_PI / n));
    std::printf("  P_n     lambda_2 = %.16f  exact %.16f  err %.3e\n", v[1], pe,
                std::fabs(v[1] - pe));

    std::vector<int> sp;
    for (int i = 1; i < n; ++i) sp.push_back(pix.index(0, i));
    spectrum(build(sp), n, v, nullptr);
    std::printf("  star    lambda_2 = %.16f  exact 1              err %.3e\n", v[1],
                std::fabs(v[1] - 1.0));

    std::vector<int> bp;
    for (int i = 0; i < n / 2; ++i)
      for (int j = n / 2; j < n; ++j) bp.push_back(pix.index(i, j));
    spectrum(build(bp), n, v, nullptr);
    std::printf("  K_a,a   lambda_2 = %.16f  exact %d       err %.3e\n", v[1], n / 2,
                std::fabs(v[1] - n / 2.0));
  }

  Rng rng(seed);
  double worstRes = 0.0;
  for (int t = 0; t < 8; ++t) {
    const int m = n - 1 + (int)(rng.next() % (uint64_t)(pix.numPairs() - n + 1));
    worstRes = std::max(worstRes, maxResidual(Graph::randomConnected(pix, m, rng), n));
  }
  std::printf("  worst eigenpair residual over 8 random graphs: %.3e\n", worstRes);

  // ---- LAYER 2: probe decisions vs dense truth -------------------------
  std::printf("\n== probe vs dense, every 1-swap (n=%d, %d graphs) ==\n", n, trials);
  long tested = 0, falseNeg = 0, falsePos = 0;
  double worstVal = 0.0, worstFN = 0.0;
  for (int t = 0; t < trials; ++t) {
    const int m = n - 1 + (int)(rng.next() % (uint64_t)(pix.numPairs() - n + 1));
    const Graph g = Graph::randomConnected(pix, m, rng);
    std::vector<double> vals, vecs;
    spectrum(g, n, vals, &vecs);
    const double lam2 = vals[1];
    if (!(lam2 > 0.0)) continue;
    // Transposed access: the probe wants vecs[u*n + i] as row u of V.
    std::vector<double> alpha(n), beta(n);
    for (int a = 0; a < pix.numPairs(); ++a) {
      if (g.has(a)) continue;
      for (int r = 0; r < pix.numPairs(); ++r) {
        if (!g.has(r)) continue;
        const int au = pix.i(a), av = pix.j(a);
        const int ru = pix.i(r), rv = pix.j(r);
        for (int i = 0; i < n; ++i) {
          alpha[i] = vecs[static_cast<size_t>(au) * n + i] -
                     vecs[static_cast<size_t>(av) * n + i];
          beta[i] = vecs[static_cast<size_t>(ru) * n + i] -
                    vecs[static_cast<size_t>(rv) * n + i];
        }
        Graph h = g;
        h.applySwap(Swap{a, r});
        std::vector<double> hv;
        spectrum(h, n, hv, nullptr);
        const double truth = hv[1];

        const double probeVal = eig::lam2AfterSwap(vals, alpha, beta);
        worstVal = std::max(worstVal, std::fabs(probeVal - truth));

        // Match production's machine-derived pole separation.  This is a
        // property of binary64 arithmetic, not a fitted acceptance margin.
        const double resolution =
            std::sqrt(std::numeric_limits<double>::epsilon()) *
            std::max(1.0, std::fabs(lam2));
        const double thr = std::nextafter(
            lam2 + resolution, std::numeric_limits<double>::infinity());
        const bool says = eig::lam2AfterSwapAtLeast(vals, alpha, beta, thr);
        const bool is = truth >= thr;
        ++tested;
        if (!says && is) {
          ++falseNeg;
          const double margin = truth - lam2;
          worstFN = std::max(worstFN, margin);
        }
        if (says && !is) ++falsePos;
      }
    }
  }
  std::printf("  swaps tested                : %ld\n", tested);
  std::printf("  worst |probe value - dense| : %.3e\n", worstVal);
  std::printf("  false NEGATIVES (lost moves): %ld  (%.4f%%), worst margin %.3e\n",
              falseNeg,
              100.0 * (double)falseNeg / (double)std::max(1L, tested),
              worstFN);
  std::printf("  false POSITIVES (wasted eval): %ld  (%.4f%%)\n", falsePos,
              100.0 * (double)falsePos / (double)std::max(1L, tested));
  return 0;
}

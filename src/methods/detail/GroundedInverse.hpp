#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "Bitset.hpp"
#include "PairIndex.hpp"

namespace algconn::detail {

// Inverse of the grounded Laplacian, maintained under edge ADDITION.
//
// Port of `IncrementalState` / `spectral.grounded_*` from the CDC 2026
// companion code (Zhu). Deleting row/column `ground` from L of a connected
// graph leaves an SPD matrix Lg; effective resistances read directly off
// Lg^{-1}, and adding an edge is a rank-one update of Lg, hence a
// Sherman-Morrison update of the inverse.
//
// Cost: one O(n^3) factorization up front, then O(n^2) per added edge instead
// of the O(n^3) a fresh inverse would take. The reference re-forms the exact
// inverse every `exactResetEvery` steps to stop rank-one updates drifting;
// that is kept, so the amortized cost stays O(n^3 / 32 + n^2) per step.
//
// This class performs no eigensolve, so callers must still bound its work
// between charged solves. Production OURS passes exactResetEvery=0 and
// ridge=0: proposal scans factor once and read the inverse, while depth paths
// use the strict no-refactor updates below. This is part of its O(n^3)
// per-budget-segment proof; the configurable defaults serve other methods.
class GroundedInverse {
 public:
  // `ridge` matches the reference's 1e-10 stabilizer on Lg.
  GroundedInverse(const PairIndex& pix, const Bitset& e, int ground = 0,
                  int exactResetEvery = 32, double ridge = 1e-10)
      : pix_(&pix), n_(pix.n()), ground_(ground), resetEvery_(exactResetEvery), ridge_(ridge) {
    recompute(e);
  }

  // Effective resistance between u and v, clamped at 0 as the reference does.
  double effectiveResistance(int u, int v) const {
    if (u == v) return 0.0;
    const int d = n_ - 1;
    if (u == ground_) { const int vi = idx(v); return std::max(inv_[vi * d + vi], 0.0); }
    if (v == ground_) { const int ui = idx(u); return std::max(inv_[ui * d + ui], 0.0); }
    const int ui = idx(u), vi = idx(v);
    return std::max(inv_[ui * d + ui] + inv_[vi * d + vi] - 2.0 * inv_[ui * d + vi], 0.0);
  }

  // TRANSFER resistance between two pairs: b_1^T L^+ b_2 with
  // b_i = e_u - e_v. The grounded inverse differs from L^+ by a constant
  // shift, which cancels in this double difference because both edge vectors
  // are orthogonal to 1. This is the off-diagonal of the resistance Gram --
  // the term the rank-two secular determinant needs and effective resistance
  // alone cannot express.
  double transfer(int u1, int v1, int u2, int v2) const {
    const int d = n_ - 1;
    auto e = [&](int a, int b) -> double {
      if (a == ground_ || b == ground_) return 0.0;
      return inv_[idx(a) * d + idx(b)];
    };
    return e(u1, u2) - e(u1, v2) - e(v1, u2) + e(v1, v2);
  }

  // Record that edge (u,v) has been added to `e` (caller updates `e` itself).
  void addEdge(int u, int v, const Bitset& e) {
    if (!shermanMorrison(u, v, +1.0)) recompute(e);
    if (resetEvery_ > 0 && ++sinceReset_ >= resetEvery_) {
      recompute(e);
      sinceReset_ = 0;
    }
  }

  // Record that edge (u,v) has been REMOVED from `e`. Deleting an edge is a
  // rank-one DOWNdate of the grounded Laplacian, so the inverse update is the
  // same formula with the opposite sign, and its denominator is
  //     1 - g^T inv g = 1 - R_eff(u,v).
  // For an edge of unit conductance R_eff <= 1 always, with equality exactly
  // when the edge is a BRIDGE -- all current must cross it, there being no
  // parallel path. The numerical guard and the connectivity guard are
  // therefore the same test, and it is free.
  void removeEdge(int u, int v, const Bitset& e) {
    if (!shermanMorrison(u, v, -1.0)) recompute(e);
    if (resetEvery_ > 0 && ++sinceReset_ >= resetEvery_) {
      recompute(e);
      sinceReset_ = 0;
    }
  }

  // Strict O(n^2) forms used by eigensolve-order search. They never hide an
  // O(n^3) refactorization: an invalid finite-precision denominator leaves the
  // inverse unchanged and reports failure, so the caller can abandon that
  // free path. In exact arithmetic a connected completed swap cannot fail.
  bool tryAddEdge(int u, int v) {
    return shermanMorrison(u, v, +1.0);
  }
  bool tryRemoveEdge(int u, int v) {
    return shermanMorrison(u, v, -1.0);
  }

  // True iff removing the present edge (u,v) would disconnect the graph.
  // Reads off the inverse in O(1); see removeEdge.
  bool isBridge(int u, int v, double tol = 1e-9) const {
    return effectiveResistance(u, v) >= 1.0 - tol;
  }

 private:
  int idx(int node) const { return node < ground_ ? node : node - 1; }

  // g = grounded incidence vector of (u,v): +1 at u, -1 at v, ground dropped.
  void incidence(int u, int v, std::vector<double>& g) const {
    g.assign(n_ - 1, 0.0);
    if (u != ground_) g[idx(u)] += 1.0;
    if (v != ground_) g[idx(v)] -= 1.0;
  }

  // sign = +1 for an added edge, -1 for a removed one.
  bool shermanMorrison(int u, int v, double sign) {
    const int d = n_ - 1;
    incidence(u, v, gbuf_);
    // x = inv * g
    xbuf_.assign(d, 0.0);
    for (int a = 0; a < d; ++a) {
      double s = 0.0;
      const double* row = &inv_[a * d];
      for (int b = 0; b < d; ++b) s += row[b] * gbuf_[b];
      xbuf_[a] = s;
    }
    double quad = 0.0;
    for (int a = 0; a < d; ++a) quad += gbuf_[a] * xbuf_[a];
    const double denom = 1.0 + sign * quad;
    if (!(denom > 0.0) || !std::isfinite(denom)) return false;
    // inv -= sign * (x x^T)/denom: addition subtracts, removal adds back.
    const double inv_denom = sign / denom;
    // inv -= (x x^T) / denom, symmetric so only the upper triangle is formed.
    for (int a = 0; a < d; ++a) {
      const double xa = xbuf_[a] * inv_denom;
      double* row = &inv_[a * d];
      for (int b = a; b < d; ++b) {
        const double val = xa * xbuf_[b];
        row[b] -= val;
        if (b != a) inv_[b * d + a] = row[b];
      }
    }
    return true;
  }

  // Exact inverse of Lg + ridge*I by Cholesky. Lg is SPD whenever the graph is
  // connected, which every caller guarantees by keeping a spanning tree.
  void recompute(const Bitset& e) {
    const int d = n_ - 1;
    std::vector<double> a(static_cast<size_t>(d) * d, 0.0);
    std::vector<int> deg(n_, 0);
    for (int k = 0; k < pix_->numPairs(); ++k) {
      if (!e.test(k)) continue;
      const int i = pix_->i(k), j = pix_->j(k);
      ++deg[i];
      ++deg[j];
      if (i == ground_ || j == ground_) continue;
      a[static_cast<size_t>(idx(i)) * d + idx(j)] = -1.0;
      a[static_cast<size_t>(idx(j)) * d + idx(i)] = -1.0;
    }
    for (int v = 0; v < n_; ++v) {
      if (v == ground_) continue;
      a[static_cast<size_t>(idx(v)) * d + idx(v)] = deg[v] + ridge_;
    }
    choleskyInverse(a, d);
    inv_.swap(a);
  }

  // In-place SPD inverse: A = L L^T, invert L, then A^{-1} = L^{-T} L^{-1}.
  static void choleskyInverse(std::vector<double>& a, int d) {
    for (int j = 0; j < d; ++j) {
      double s = a[static_cast<size_t>(j) * d + j];
      for (int k = 0; k < j; ++k) { const double v = a[static_cast<size_t>(j) * d + k]; s -= v * v; }
      if (!(s > 0.0) || !std::isfinite(s))
        throw std::runtime_error("grounded Laplacian is not numerically SPD");
      const double ljj = std::sqrt(s);
      a[static_cast<size_t>(j) * d + j] = ljj;
      for (int i = j + 1; i < d; ++i) {
        double t = a[static_cast<size_t>(i) * d + j];
        for (int k = 0; k < j; ++k)
          t -= a[static_cast<size_t>(i) * d + k] * a[static_cast<size_t>(j) * d + k];
        a[static_cast<size_t>(i) * d + j] = t / ljj;
      }
    }
    // Invert the lower triangle in place.
    for (int j = 0; j < d; ++j) {
      a[static_cast<size_t>(j) * d + j] = 1.0 / a[static_cast<size_t>(j) * d + j];
      for (int i = j + 1; i < d; ++i) {
        double t = 0.0;
        for (int k = j; k < i; ++k)
          t -= a[static_cast<size_t>(i) * d + k] * a[static_cast<size_t>(k) * d + j];
        a[static_cast<size_t>(i) * d + j] = t / a[static_cast<size_t>(i) * d + i];
      }
    }
    // A^{-1} = L^{-T} L^{-1}, forming the symmetric result.
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int k = i; k < d; ++k)
          s += a[static_cast<size_t>(k) * d + i] * a[static_cast<size_t>(k) * d + j];
        a[static_cast<size_t>(i) * d + j] = s;
      }
    }
    for (int i = 0; i < d; ++i)
      for (int j = i + 1; j < d; ++j)
        a[static_cast<size_t>(i) * d + j] = a[static_cast<size_t>(j) * d + i];
  }

  const PairIndex* pix_;
  int n_, ground_, resetEvery_, sinceReset_ = 0;
  double ridge_;
  std::vector<double> inv_, gbuf_, xbuf_;
};

}  // namespace algconn::detail

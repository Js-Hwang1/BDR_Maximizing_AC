#include "Eigen.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace algconn::eig {
namespace {

inline double hypot2(double a, double b) {
  const double x = std::fabs(a), y = std::fabs(b);
  if (x > y) { const double r = y / x; return x * std::sqrt(1.0 + r * r); }
  if (y == 0.0) return 0.0;
  const double r = x / y;
  return y * std::sqrt(1.0 + r * r);
}

// Householder reduction of a symmetric matrix to tridiagonal form.
// d <- diagonal, e <- sub-diagonal, a <- accumulated transform.
void tred2(std::vector<double>& a, int n, std::vector<double>& d,
           std::vector<double>& e) {
  auto A = [&](int r, int c) -> double& { return a[static_cast<size_t>(r) * n + c]; };

  for (int j = 0; j < n; ++j) d[j] = A(n - 1, j);

  for (int i = n - 1; i > 0; --i) {
    double scale = 0.0, h = 0.0;
    for (int k = 0; k < i; ++k) scale += std::fabs(d[k]);
    if (scale == 0.0) {
      e[i] = d[i - 1];
      for (int j = 0; j < i; ++j) { d[j] = A(i - 1, j); A(i, j) = 0.0; A(j, i) = 0.0; }
    } else {
      for (int k = 0; k < i; ++k) { d[k] /= scale; h += d[k] * d[k]; }
      double f = d[i - 1];
      double g = (f > 0.0) ? -std::sqrt(h) : std::sqrt(h);
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (int j = 0; j < i; ++j) e[j] = 0.0;

      for (int j = 0; j < i; ++j) {
        f = d[j];
        A(j, i) = f;
        g = e[j] + A(j, j) * f;
        for (int k = j + 1; k <= i - 1; ++k) {
          g += A(k, j) * d[k];
          e[k] += A(k, j) * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (int j = 0; j < i; ++j) { e[j] /= h; f += e[j] * d[j]; }
      const double hh = f / (h + h);
      for (int j = 0; j < i; ++j) e[j] -= hh * d[j];
      for (int j = 0; j < i; ++j) {
        f = d[j];
        g = e[j];
        for (int k = j; k <= i - 1; ++k) A(k, j) -= (f * e[k] + g * d[k]);
        d[j] = A(i - 1, j);
        A(i, j) = 0.0;
      }
    }
    d[i] = h;
  }

  // Accumulate the transformation.
  for (int i = 0; i < n - 1; ++i) {
    A(n - 1, i) = A(i, i);
    A(i, i) = 1.0;
    const double h = d[i + 1];
    if (h != 0.0) {
      for (int k = 0; k <= i; ++k) d[k] = A(k, i + 1) / h;
      for (int j = 0; j <= i; ++j) {
        double g = 0.0;
        for (int k = 0; k <= i; ++k) g += A(k, i + 1) * A(k, j);
        for (int k = 0; k <= i; ++k) A(k, j) -= g * d[k];
      }
    }
    for (int k = 0; k <= i; ++k) A(k, i + 1) = 0.0;
  }
  for (int j = 0; j < n; ++j) { d[j] = A(n - 1, j); A(n - 1, j) = 0.0; }
  A(n - 1, n - 1) = 1.0;
  e[0] = 0.0;
}

// Implicit-shift QL on the tridiagonal (d, e); a holds the accumulated
// transform on entry and the eigenvectors on exit.
void tql2(std::vector<double>& a, int n, std::vector<double>& d,
          std::vector<double>& e) {
  auto A = [&](int r, int c) -> double& { return a[static_cast<size_t>(r) * n + c]; };

  for (int i = 1; i < n; ++i) e[i - 1] = e[i];
  e[n - 1] = 0.0;

  double f = 0.0, tst1 = 0.0;
  const double eps = std::numeric_limits<double>::epsilon();

  for (int l = 0; l < n; ++l) {
    tst1 = std::max(tst1, std::fabs(d[l]) + std::fabs(e[l]));
    int mm = l;
    while (mm < n) {
      if (std::fabs(e[mm]) <= eps * tst1) break;
      ++mm;
    }
    if (mm > l) {
      // One iteration must retire meaningful mantissa information. Refuse to
      // spin beyond the number of representable significand bits: this is a
      // machine-derived convergence guard, not a fitted iteration limit.
      int iterations = 0;
      for (;;) {
        if (iterations++ == std::numeric_limits<double>::digits)
          throw std::runtime_error("symmetric QL failed to converge");
        double g = d[l];
        double p = (d[l + 1] - g) / (2.0 * e[l]);
        double r = hypot2(p, 1.0);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const double dl1 = d[l + 1];
        double h = g - d[l];
        for (int i = l + 2; i < n; ++i) d[i] -= h;
        f += h;

        p = d[mm];
        double c = 1.0, c2 = c, c3 = c;
        const double el1 = e[l + 1];
        double s = 0.0, s2 = 0.0;
        for (int i = mm - 1; i >= l; --i) {
          c3 = c2; c2 = c; s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot2(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          for (int k = 0; k < n; ++k) {
            h = A(k, i + 1);
            A(k, i + 1) = s * A(k, i) + c * h;
            A(k, i) = c * A(k, i) - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
        if (!(std::fabs(e[l]) > eps * tst1)) break;
      }
    }
    d[l] += f;
    e[l] = 0.0;
  }

  // Sort ascending, carrying the eigenvectors.
  for (int i = 0; i < n - 1; ++i) {
    int k = i;
    double p = d[i];
    for (int j = i + 1; j < n; ++j)
      if (d[j] < p) { k = j; p = d[j]; }
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      for (int j = 0; j < n; ++j) std::swap(a[static_cast<size_t>(j) * n + i],
                                            a[static_cast<size_t>(j) * n + k]);
    }
  }
}

}  // namespace

namespace {

// Implicit-shift QL on the tridiagonal, eigenvalues only.
void tql2NoVec(int n, std::vector<double>& d, std::vector<double>& e) {
  for (int i = 1; i < n; ++i) e[i - 1] = e[i];
  e[n - 1] = 0.0;
  double f = 0.0, tst1 = 0.0;
  const double eps = std::numeric_limits<double>::epsilon();
  for (int l = 0; l < n; ++l) {
    tst1 = std::max(tst1, std::fabs(d[l]) + std::fabs(e[l]));
    int mm = l;
    while (mm < n && std::fabs(e[mm]) > eps * tst1) ++mm;
    if (mm > l) {
      int iterations = 0;
      for (;;) {
        if (iterations++ == std::numeric_limits<double>::digits)
          throw std::runtime_error("symmetric values-only QL failed to converge");
        double g = d[l];
        double p = (d[l + 1] - g) / (2.0 * e[l]);
        double r = hypot2(p, 1.0);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const double dl1 = d[l + 1];
        double h = g - d[l];
        for (int i = l + 2; i < n; ++i) d[i] -= h;
        f += h;
        p = d[mm];
        double c = 1.0, c2 = c, c3 = c;
        const double el1 = e[l + 1];
        double s = 0.0, s2 = 0.0;
        for (int i = mm - 1; i >= l; --i) {
          c3 = c2; c2 = c; s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot2(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
        if (!(std::fabs(e[l]) > eps * tst1)) break;
      }
    }
    d[l] += f;
    e[l] = 0.0;
  }
  std::sort(d.begin(), d.end());
}

}  // namespace

void symmetricValues(std::vector<double>& a, int n, std::vector<double>& vals) {
  // Reuse the validated tred2 -- its accumulation phase is what finalises the
  // diagonal, so it cannot simply be dropped -- and skip only the QL
  // eigenvector rotations, which are the larger of the two costs.
  vals.assign(n, 0.0);
  std::vector<double> e(n, 0.0);
  tred2(a, n, vals, e);
  tql2NoVec(n, vals, e);
}

int lambda2Multiplicity(const std::vector<double>& vals, double rtol) {
  const int n = static_cast<int>(vals.size());
  if (n < 2) return 0;
  const double norm = std::max(1.0, std::fabs(vals[n - 1]));
  const double atol = rtol * norm;
  const double lam2 = vals[1];
  int k = 0;
  for (int i = 1; i < n; ++i)
    if (std::fabs(vals[i] - lam2) <= atol) ++k; else break;
  return k;
}

void symmetric(std::vector<double>& a, int n, std::vector<double>& vals,
               bool wantVectors) {
  vals.assign(n, 0.0);
  std::vector<double> e(n, 0.0);
  tred2(a, n, vals, e);
  tql2(a, n, vals, e);
  if (!wantVectors) a.clear();
}

}  // namespace algconn::eig

namespace algconn::eig {

namespace {

// Count of eigenvalues of L' = L + a a^T - b b^T strictly below mu (mu not an
// eigenvalue of L): negL(mu) + neg( [[-1 - Saa, -Sab], [-Sab, 1 - Sbb]] ) - 1,
// from inertia additivity on [[L - mu, U], [U^T, -W^{-1}]], U = [b_a b_r],
// W = diag(1, -1). One O(n) pass.
int swappedCountBelow(const std::vector<double>& vals, const std::vector<double>& a,
                      const std::vector<double>& b, double mu) {
  const int n = static_cast<int>(vals.size());
  int cnt = 0;
  double saa = 0.0, sbb = 0.0, sab = 0.0;
  for (int i = 0; i < n; ++i) {
    if (vals[i] < mu) ++cnt;
    const double d = vals[i] - mu;
    saa += a[i] * a[i] / d;
    sbb += b[i] * b[i] / d;
    sab += a[i] * b[i] / d;
  }
  const double p = -1.0 - saa, s = 1.0 - sbb, q = -sab;
  const double det = p * s - q * q, tr = p + s;
  int neg2;
  if (det > 0.0) neg2 = tr < 0.0 ? 2 : 0;
  else if (det < 0.0) neg2 = 1;
  else neg2 = tr < 0.0 ? 1 : 0;
  return cnt + neg2 - 1;
}

}  // namespace

double lam2AfterSwap(const std::vector<double>& vals, const std::vector<double>& a,
                     const std::vector<double>& b) {
  const int n = static_cast<int>(vals.size());
  auto countBelow = [&](double mu) { return swappedCountBelow(vals, a, b, mu); };

  // lambda_2(L') = inf{mu > 0 : #below(mu) >= 2}; the zero eigenvalue always
  // survives a swap (L' 1 = 0). Interlacing brackets it by lambda_4(L).
  double scale = 1.0;
  for (double value : vals) scale = std::max(scale, std::fabs(value));
  double lo = std::numeric_limits<double>::epsilon() * scale;
  double hi = std::nextafter(
      n > 3 ? vals[3] : vals[n - 1],
      std::numeric_limits<double>::infinity());
  // A binary64 interval cannot be refined more than its significand length
  // without one endpoint repeating. The bound is machine-derived, not a
  // numerical tuning parameter.
  for (int it = 0; it <= std::numeric_limits<double>::digits; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (mid == lo || mid == hi) break;
    if (countBelow(mid) >= 2) hi = mid;
    else lo = mid;
  }
  return 0.5 * (lo + hi);
}

bool lam2AfterSwapAtLeast(const std::vector<double>& vals, const std::vector<double>& a,
                          const std::vector<double>& b, double thr) {
  return swappedCountBelow(vals, a, b, thr) <= 1;
}

int swapCountBelow(const std::vector<double>& vals, const std::vector<double>& a,
                   const std::vector<double>& b, double thr) {
  return swappedCountBelow(vals, a, b, thr);
}

ShiftedSwapInertia::ShiftedSwapInertia(const std::vector<double>& vals,
                                       const std::vector<double>& vecs,
                                       double thr)
    : n_(static_cast<int>(vals.size())),
      resolvent_(static_cast<size_t>(n_) * n_, 0.0) {
  for (double value : vals) {
    if (value < thr) ++anchorCountBelow_;
    if (value == thr)
      throw std::invalid_argument(
          "shifted swap inertia threshold is an anchor eigenvalue");
  }
  for (int mode = 0; mode < n_; ++mode) {
    const double inverseShift = 1.0 / (vals[mode] - thr);
    for (int row = 0; row < n_; ++row) {
      const double weighted =
          vecs[static_cast<size_t>(row) * n_ + mode] * inverseShift;
      for (int column = 0; column < n_; ++column)
        resolvent_[static_cast<size_t>(row) * n_ + column] +=
            weighted * vecs[static_cast<size_t>(column) * n_ + mode];
    }
  }
}

int ShiftedSwapInertia::countBelow(int addU, int addV,
                                   int removeU, int removeV) const {
  return countBelow(projectPair(addU, addV),
                    projectPair(removeU, removeV));
}

int ShiftedSwapInertia::countBelow(
    const PairProjection& add, const PairProjection& remove) const {
  const double sab = pairTransfer(add.u, add.v, remove.u, remove.v);
  const double p = -1.0 - add.quadratic;
  const double s = 1.0 - remove.quadratic;
  const double q = -sab;
  const double determinant = p * s - q * q;
  const double trace = p + s;
  int negativeCorrection = 0;
  if (determinant > 0.0)
    negativeCorrection = trace < 0.0 ? 2 : 0;
  else if (determinant < 0.0)
    negativeCorrection = 1;
  else
    negativeCorrection = trace < 0.0 ? 1 : 0;
  return anchorCountBelow_ + negativeCorrection - 1;
}

}  // namespace algconn::eig

namespace algconn::eig {

bool lam2AfterEditAtLeast(const std::vector<double>& vals, const std::vector<double>& coords,
                          const std::vector<int>& signs, double thr) {
  const int n = static_cast<int>(vals.size());
  const int p = static_cast<int>(signs.size());
  // Count eigenvalues of L below thr, and build S(thr) in the same pass.
  int cnt = 0;
  std::vector<double> S(static_cast<size_t>(p) * p, 0.0);
  for (int t = 0; t < n; ++t) {
    if (vals[t] < thr) ++cnt;
    const double inv = 1.0 / (vals[t] - thr);
    for (int i = 0; i < p; ++i) {
      const double ci = coords[static_cast<size_t>(i) * n + t] * inv;
      for (int j = i; j < p; ++j)
        S[static_cast<size_t>(i) * p + j] += ci * coords[static_cast<size_t>(j) * n + t];
    }
  }
  // G = -W^{-1} - S with W = diag(signs), W^{-1} = W.
  std::vector<double> G(static_cast<size_t>(p) * p);
  int negW = 0;
  for (int i = 0; i < p; ++i) {
    if (signs[i] > 0) ++negW;  // -W^{-1} has a -1 exactly at the +1 signs
    for (int j = i; j < p; ++j) {
      double g = -S[static_cast<size_t>(i) * p + j];
      if (i == j) g -= static_cast<double>(signs[i]);
      G[static_cast<size_t>(i) * p + j] = g;
      G[static_cast<size_t>(j) * p + i] = g;
    }
  }
  std::vector<double> gvals;
  symmetricValues(G, p, gvals);
  int negG = 0;
  for (double v : gvals) negG += v < 0.0;
  return cnt + negG - negW <= 1;
}

}  // namespace algconn::eig

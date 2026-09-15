#include "MomentBounds.hpp"

#include <algorithm>
#include <cmath>

namespace algconn::mb {
namespace {

// A measure on [0,inf) with moments (n0,n1,n2,n3) exists iff both Hankel
// matrices [[n0,n1],[n1,n2]] and [[n1,n2],[n2,n3]] are PSD. For 2x2 symmetric
// matrices that is: diagonal >= 0 and determinant >= 0 -- no eigensolver.
inline bool stieltjesFeasible(double t, double k, double m1, double m2, double m3) {
  const double n0 = k;
  const double n1 = m1 - t * k;
  const double n2 = m2 - 2 * t * m1 + t * t * k;
  const double n3 = m3 - 3 * t * m2 + 3 * t * t * m1 - t * t * t * k;
  const double eps = -1e-9;
  if (n1 < eps || n2 < eps || n3 < eps) return false;
  if (n0 * n2 - n1 * n1 < eps) return false;
  if (n1 * n3 - n2 * n2 < eps) return false;
  return true;
}

double bisectThree(double k, double m1, double m2, double m3) {
  double lo = 0.0, hi = m1 / k;
  for (int i = 0; i < 100; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (stieltjesFeasible(mid, k, m1, m2, m3)) lo = mid; else hi = mid;
  }
  return lo;
}

double bisectTwo(double k, double m1, double m2) {
  const double mu = m1 / k;
  const double var = std::max(m2 / k - mu * mu, 0.0);
  return mu - std::sqrt(var) / std::sqrt(k - 1.0);
}

}  // namespace

double fiedler(int n, long m) {
  // lambda_2 <= kappa_v <= delta_min <= floor(2m/n) holds only for NON-COMPLETE
  // graphs: K_n has lambda_2 = n while floor(2m/n) = n-1, so the chain fails
  // there and the complete case must be returned exactly.
  const long full = (long)n * (n - 1) / 2;
  if (m >= full) return (double)n;
  return std::min((double)(n - 2), std::floor(2.0 * m / n));
}

double twoMoment(int n, long m, const std::vector<int>& deg) {
  double s2 = 0;
  for (int d : deg) s2 += (double)d * d;
  return bisectTwo(n - 1.0, 2.0 * m, s2 + 2.0 * m);
}

double threeMoment(int n, long m, const std::vector<int>& deg, long tri) {
  double s2 = 0, s3 = 0;
  for (int d : deg) { s2 += (double)d * d; s3 += (double)d * d * d; }
  return bisectThree(n - 1.0, 2.0 * m, s2 + 2.0 * m, s3 + 3.0 * s2 - 6.0 * tri);
}

namespace {
// Maximise a moment bound over two-level degree sequences: p vertices of degree
// D and n-p of degree d, with pD + (n-p)d = 2m. Two-level sequences are
// extremal for moment problems; relaxing D,d to reals keeps the result an
// upper bound.
template <class F>
double maximiseOverDegrees(int n, long m, F&& evaluate) {
  double best = 0.0;
  for (int p = 1; p < n; ++p) {
    for (int D = 1; D <= n - 1; ++D) {
      const double rem = 2.0 * m - (double)p * D;
      if (rem < 0) continue;
      const double d = rem / (n - p);
      if (d < 0.0 || d > n - 1) continue;
      const double s2 = p * (double)D * D + (n - p) * d * d;
      const double s3 = p * (double)D * D * D + (n - p) * d * d * d;
      best = std::max(best, evaluate(s2, s3));
    }
  }
  return best;
}
}  // namespace

double twoMomentNM(int n, long m) {
  return maximiseOverDegrees(n, m, [&](double s2, double) {
    return bisectTwo(n - 1.0, 2.0 * m, s2 + 2.0 * m);
  });
}

double threeMomentNM(int n, long m) {
  return maximiseOverDegrees(n, m, [&](double s2, double s3) {
    // T = 0 maximises tr(L^3) and hence the bound: conservative, so valid.
    return bisectThree(n - 1.0, 2.0 * m, s2 + 2.0 * m, s3 + 3.0 * s2);
  });
}

double best(int n, long m) {
  return std::min(fiedler(n, m), std::min(twoMomentNM(n, m), threeMomentNM(n, m)));
}

}  // namespace algconn::mb

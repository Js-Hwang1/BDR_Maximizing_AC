#pragma once
#include <vector>

namespace algconn::eig {

// Symmetric eigendecomposition, self-contained (no BLAS/LAPACK).
//
// Householder reduction to tridiagonal form (tred2) followed by implicit-shift
// QL (tql2). Eigenvalues are returned ASCENDING, so lambda_2 is vals[1].
//
// a       row-major n*n symmetric matrix; overwritten with eigenvectors
//         (column k of the result is vals[k]'s eigenvector) if wantVectors.
// vals    resized to n, ascending.
//
// Degenerate eigenvalues are the normal case at optima here, so the full
// spectrum is always produced rather than a single "the" Fiedler vector.
void symmetric(std::vector<double>& a, int n, std::vector<double>& vals,
               bool wantVectors);

// Eigenvalues only. Skips accumulating the Householder transform and the
// eigenvector rotations in QL, which is the dominant cost -- roughly 3-5x
// faster than the vector-producing path. Most oracle calls only ever need
// lambda_2 (a swap is verified by its eigenvalue), so this is the hot path.
void symmetricValues(std::vector<double>& a, int n, std::vector<double>& vals);

// Multiplicity of vals[1] (i.e. of lambda_2), clustering with a tolerance tied
// to the spectral norm. Degenerate lambda_2 is the NORMAL case at optima here,
// so this is a first-class quantity, not a corner case.
int lambda2Multiplicity(const std::vector<double>& vals, double rtol = 1e-9);

// Exact-arithmetic lambda_2 identity for L' = L + b_a b_a^T - b_r b_r^T,
// evaluated from a computed full eigendecomposition of L without forming or
// solving L'. a and b are the
// perturbation vectors expressed in the eigenbasis (a_i = v_i . b_a, i.e. the
// per-eigenvector coordinates); vals ascending with vals[0] = 0.
//
// Haynsworth's inertia identity reduces the eigenvalue count of L' below mu
// to the count for L plus the negative inertia of a 2x2 matrix of spectral
// sums -- O(n) per probe -- and bisection on that count pins lambda_2(L')
// between machine-precision brackets. The counting form is chosen over a
// root-finder deliberately: it is immune to the clustered eigenvalues that
// are the normal case here. Assumes L' still has 0 as its smallest eigenvalue
// (true for every Laplacian-to-Laplacian swap) and that the caller has
// screened connectivity. Cost: ~60 probes * O(n).
double lam2AfterSwap(const std::vector<double>& vals, const std::vector<double>& a,
                     const std::vector<double>& b);

// Decision form of the same identity, ONE O(n) probe: does the swapped graph
// satisfy lambda_2(L') >= thr? (i.e., fewer than two eigenvalues of L' lie
// below thr). This is what a climb needs per candidate -- the exact value is
// only ever needed for the single accepted move, and the paid verification
// call supplies it anyway.
bool lam2AfterSwapAtLeast(const std::vector<double>& vals, const std::vector<double>& a,
                          const std::vector<double>& b, double thr);

// The underlying count: how many eigenvalues of the swapped Laplacian lie
// strictly below thr. count <= 1 is the improvement decision above; count
// <= 2 is the free NECESSARY condition for a two-swap through this
// intermediate to clear thr (lambda_3 of the intermediate must exceed thr).
int swapCountBelow(const std::vector<double>& vals, const std::vector<double>& a,
                   const std::vector<double>& b, double thr);

// Shared-threshold form of the same rank-two inertia calculation. Building
// K(thr) = V diag(1/(lambda_i-thr)) V^T costs O(n^3) once; every completed
// edge-swap count then uses four endpoint lookups per quadratic form, O(1).
// `thr` must not equal an eigenvalue in `vals`.
class ShiftedSwapInertia {
 public:
  struct PairProjection {
    int u = -1;
    int v = -1;
    double quadratic = 0.0;
  };

  ShiftedSwapInertia(const std::vector<double>& vals,
                     const std::vector<double>& vecs, double thr);
  PairProjection projectPair(int u, int v) const {
    return PairProjection{u, v, pairQuadratic(u, v)};
  }
  int countBelow(int addU, int addV, int removeU, int removeV) const;
  int countBelow(const PairProjection& add,
                 const PairProjection& remove) const;
  bool lam2AtLeast(int addU, int addV, int removeU, int removeV) const {
    return countBelow(addU, addV, removeU, removeV) <= 1;
  }
  bool lam2AtLeast(const PairProjection& add,
                   const PairProjection& remove) const {
    return countBelow(add, remove) <= 1;
  }

 private:
  double entry(int row, int column) const {
    return resolvent_[static_cast<size_t>(row) * n_ + column];
  }
  double pairQuadratic(int u, int v) const {
    return entry(u, u) + entry(v, v) - entry(u, v) - entry(v, u);
  }
  double pairTransfer(int u, int v, int x, int y) const {
    return entry(u, x) - entry(u, y) - entry(v, x) + entry(v, y);
  }

  int n_ = 0;
  int anchorCountBelow_ = 0;
  std::vector<double> resolvent_;
};

// The same decision for a COMPOSITE edit of rank p: L' = L + sum_i w_i u_i
// u_i^T with signs w_i = +1 (inserted pairs) or -1 (removed edges). `coords`
// holds the p perturbation vectors IN THE EIGENBASIS, row-major p*n (row i =
// V^T u_i). Haynsworth again: the eigenvalue count of L' below thr equals the
// count for L plus the negative inertia of the p-by-p matrix -W^{-1}-S(thr),
// S_ij = sum_t coords_i[t] coords_j[t]/(vals[t]-thr), minus the number of +1
// signs. One O(n p^2 + p^3) probe decides one selected composite edit between
// paid solves; the production search does not enumerate the radius-r ball.
bool lam2AfterEditAtLeast(const std::vector<double>& vals, const std::vector<double>& coords,
                          const std::vector<int>& signs, double thr);

}  // namespace algconn::eig

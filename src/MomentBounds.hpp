#pragma once
#include <vector>

namespace algconn::mb {

// Certified upper bounds on lambda_2 over G_{n,m}, from SPECTRAL MOMENTS.
//
// The k = n-1 nonzero-index Laplacian eigenvalues satisfy
//   tr L   = sum_i lambda_i     = 2m                       (constant on G_{n,m})
//   tr L^2 = sum_i lambda_i^2   = sum_i d_i^2 + 2m          (degree sequence)
//   tr L^3 = sum_i lambda_i^3   = sum_i d_i^3 + 3 sum d_i^2 - 6T   (T = triangles)
// All three are COMBINATORIAL -- no eigensolve is ever required.
//
// Bounding min(lambda) given its moments is a truncated Stieltjes moment
// problem. With two moments the answer is closed form; with three it is the
// largest t for which a measure on [t, inf) with those moments exists, decided
// by two 2x2 Hankel determinants. By Chebyshev-Markov the r-moment bound is
// TIGHT exactly for r-level spectra:
//   2 moments -> {0, theta^(n-2), Theta}   : star, and K_{n/2,n/2} for even n
//   3 moments -> three-level spectra       : general complete bipartite K_{a,b}

// Classical Fiedler chain: lambda_2 <= kappa <= delta_min <= floor(2m/n),
// plus lambda_2 <= n-2 for non-complete graphs (lambda_2 = n only for K_n).
double fiedler(int n, long m);

// Two-moment bound evaluated on a GIVEN degree sequence.
double twoMoment(int n, long m, const std::vector<int>& deg);

// Three-moment bound on a given degree sequence and triangle count.
double threeMoment(int n, long m, const std::vector<int>& deg, long triangles);

// (n,m)-ONLY certificates: maximised over admissible degree sequences, so they
// bound EVERY graph in G_{n,m}. The maximisation is relaxed (real-valued
// degrees, two levels, T = 0), which can only weaken the bound and therefore
// keeps it valid. T = 0 is conservative because the bound is DECREASING in T.
double twoMomentNM(int n, long m);
double threeMomentNM(int n, long m);

// Best available certificate: the tightest of the above.
double best(int n, long m);

}  // namespace algconn::mb

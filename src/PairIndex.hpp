#pragma once
#include <cassert>
#include <cstdint>
#include <vector>

namespace algconn {

// Lexicographic index of the binom(n,2) unordered vertex pairs.
// Pair k is (i(k), j(k)) with i < j, in row-major upper-triangular order.
class PairIndex {
 public:
  explicit PairIndex(int n) : n_(n), i_(), j_() {
    assert(n >= 2);
    const int np = n * (n - 1) / 2;
    i_.reserve(np);
    j_.reserve(np);
    for (int a = 0; a < n; ++a)
      for (int b = a + 1; b < n; ++b) { i_.push_back(a); j_.push_back(b); }
  }

  int n() const { return n_; }
  int numPairs() const { return static_cast<int>(i_.size()); }
  int i(int k) const { return i_[k]; }
  int j(int k) const { return j_[k]; }

  // Index of pair (a,b); order of a,b does not matter.
  int index(int a, int b) const {
    if (a > b) { const int t = a; a = b; b = t; }
    assert(0 <= a && a < b && b < n_);
    return a * (2 * n_ - a - 1) / 2 + (b - a - 1);
  }

 private:
  int n_;
  std::vector<int> i_, j_;
};

}  // namespace algconn

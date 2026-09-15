#pragma once
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace algconn {

// Fixed-width bitset over the vertex pairs. Only what the engine needs:
// test/set/clear, popcount, and the set operations crossover uses.
class Bitset {
 public:
  Bitset() = default;
  explicit Bitset(int nbits)
      : nbits_(nbits), w_((static_cast<size_t>(nbits) + 63) / 64, 0ULL) {}

  int size() const { return nbits_; }

  bool test(int k) const {
    assert(0 <= k && k < nbits_);
    return (w_[k >> 6] >> (k & 63)) & 1ULL;
  }
  void set(int k) {
    assert(0 <= k && k < nbits_);
    w_[k >> 6] |= (1ULL << (k & 63));
  }
  void clear(int k) {
    assert(0 <= k && k < nbits_);
    w_[k >> 6] &= ~(1ULL << (k & 63));
  }

  int count() const {
    int c = 0;
    for (uint64_t x : w_) c += std::popcount(x);
    return c;
  }

  // Indices of set bits, ascending.
  std::vector<int> ones() const {
    std::vector<int> out;
    out.reserve(count());
    for (size_t wi = 0; wi < w_.size(); ++wi) {
      uint64_t x = w_[wi];
      while (x) {
        const int b = std::countr_zero(x);
        out.push_back(static_cast<int>(wi) * 64 + b);
        x &= x - 1;
      }
    }
    return out;
  }

  // Indices set in *this but not in other.
  std::vector<int> minus(const Bitset& other) const {
    assert(nbits_ == other.nbits_);
    std::vector<int> out;
    for (size_t wi = 0; wi < w_.size(); ++wi) {
      uint64_t x = w_[wi] & ~other.w_[wi];
      while (x) {
        const int b = std::countr_zero(x);
        out.push_back(static_cast<int>(wi) * 64 + b);
        x &= x - 1;
      }
    }
    return out;
  }

  bool operator==(const Bitset& o) const { return nbits_ == o.nbits_ && w_ == o.w_; }

  const std::vector<uint64_t>& words() const { return w_; }

 private:
  int nbits_ = 0;
  std::vector<uint64_t> w_;
};

}  // namespace algconn

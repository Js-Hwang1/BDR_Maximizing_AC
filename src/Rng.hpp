#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace algconn {

// xoshiro256** seeded through splitmix64. Deterministic and self-contained:
// results must reproduce bit-for-bit from the seed alone.
class Rng {
 public:
  explicit Rng(uint64_t seed) {
    uint64_t z = seed;
    for (int k = 0; k < 4; ++k) s_[k] = splitmix64(z);
  }

  uint64_t next() {
    const uint64_t r = rotl(s_[1] * 5, 7) * 9;
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
    s_[2] ^= t;     s_[3] = rotl(s_[3], 45);
    return r;
  }

  // Uniform in [0, bound). Lemire's debiased multiply-shift.
  uint32_t below(uint32_t bound) {
    uint64_t m = static_cast<uint64_t>(static_cast<uint32_t>(next())) * bound;
    uint32_t l = static_cast<uint32_t>(m);
    if (l < bound) {
      const uint32_t t = (-bound) % bound;
      int rejected = 0;
      while (l < t) {
        // Exact debiasing is a rejection sampler. Its rejection probability
        // is below one half, but do not leave a literal unbounded runtime path:
        // one machine-word worth of consecutive failures is a deterministic,
        // representation-derived failure condition, not a fitted retry count.
        if (rejected++ == std::numeric_limits<uint64_t>::digits)
          throw std::runtime_error("uniform RNG rejection failed to terminate");
        m = static_cast<uint64_t>(static_cast<uint32_t>(next())) * bound;
        l = static_cast<uint32_t>(m);
      }
    }
    return static_cast<uint32_t>(m >> 32);
  }

  // Uniform in [0,1).
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

 private:
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  static uint64_t splitmix64(uint64_t& z) {
    z += 0x9E3779B97F4A7C15ULL;
    uint64_t r = z;
    r = (r ^ (r >> 30)) * 0xBF58476D1CE4E5B9ULL;
    r = (r ^ (r >> 27)) * 0x94D049BB133111EBULL;
    return r ^ (r >> 31);
  }
  uint64_t s_[4];
};

}  // namespace algconn

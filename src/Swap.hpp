#pragma once

namespace algconn {

// The atomic rewiring move: insert pair `add`, delete pair `remove`.
// Because the two happen together, |E| is invariant by construction.
struct Swap {
  int add = -1;
  int remove = -1;
};

}  // namespace algconn

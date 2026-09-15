#pragma once
#include <string>
#include <utility>
#include <vector>

namespace algconn {

// One run of one method on one instance.
struct Result {
  std::string method;
  int n = 0;
  int m = 0;
  double lam2 = 0.0;
  std::vector<std::pair<int, int>> edges;  // exactly m entries, i < j
  long oracleCalls = 0;                    // the budget actually spent
  long oracleQueries = 0;
  double wallSec = 0.0;
  uint64_t seed = 0;
  int steps = 0;              // method-specific progress counter
  double momentBound = 0.0;   // zero-cost certified upper bound at the optimum
  // Multiplicity of lambda_2 at the returned graph. For FVClimb this is the
  // multiplicity AT THE LEAF where the climb halted -- the quantity that shows
  // the first-order Fiedler-gap rule is ill-posed exactly where it stops.
  int multiplicity = 0;
};

}  // namespace algconn

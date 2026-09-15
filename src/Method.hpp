#pragma once
#include <string>

#include "Graph.hpp"
#include "Oracle.hpp"
#include "Result.hpp"
#include "Rng.hpp"

namespace algconn {

// Every method receives the same three things: a starting graph, an oracle
// carrying the budget, and a generator. That is the paper's comparison
// protocol -- same per-seed initialization, same swap neighborhood, same B --
// enforced by the interface rather than by discipline.
class Method {
 public:
  virtual ~Method() = default;
  virtual std::string name() const = 0;
  virtual Result run(Graph start, Oracle& oracle, Rng& rng) const = 0;
};

}  // namespace algconn

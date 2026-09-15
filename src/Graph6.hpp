#pragma once
#include <string>

#include "Graph.hpp"

namespace algconn {

// graph6 encoding (McKay). Canonical, compact, and still readable in ten years
// -- the format the results must be stored in so structure can be analysed
// downstream rather than just aggregate lambda_2.
//
// Bit order is graph6's own: for j = 1..n-1, for i = 0..j-1, the bit for (i,j).
std::string toGraph6(const Graph& g);

}  // namespace algconn

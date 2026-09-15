#include "Graph6.hpp"

#include <stdexcept>
#include <vector>

namespace algconn {

std::string toGraph6(const Graph& g) {
  const int n = g.n();
  if (n < 1 || n > 62)
    throw std::invalid_argument("toGraph6 supports 1 <= n <= 62");

  std::string out;
  out.push_back(static_cast<char>(n + 63));

  // Upper triangle in graph6 order: column by column.
  std::vector<int> bits;
  bits.reserve(static_cast<size_t>(n) * (n - 1) / 2);
  for (int j = 1; j < n; ++j)
    for (int i = 0; i < j; ++i)
      bits.push_back(g.has(g.pix().index(i, j)) ? 1 : 0);

  for (size_t p = 0; p < bits.size(); p += 6) {
    int v = 0;
    for (int k = 0; k < 6; ++k) {
      v <<= 1;
      if (p + k < bits.size()) v |= bits[p + k];
    }
    out.push_back(static_cast<char>(v + 63));
  }
  return out;
}

}  // namespace algconn

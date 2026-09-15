// lambda_2 of the shared start graph G0, per seed.
//
// Every method in a sweep begins at the same per-seed G0, so the first oracle
// call of every run evaluates the same graph: the value at B = 1 is lambda_2(G0),
// identical across methods. G0 is a deterministic function of (n, m, seed), so
// it can be reconstructed here rather than recovered from a tape -- which is
// what makes the left edge of a budget curve free to compute instead of
// something to re-run a sweep for.
#include <cstdio>
#include <string>

#include "Graph.hpp"
#include "Oracle.hpp"
#include "PairIndex.hpp"
#include "Rng.hpp"

using namespace algconn;

static uint64_t cellSeed(int n, int m, uint64_t s) {
  uint64_t h = 1469598103934665603ULL;
  for (uint64_t x : {(uint64_t)n, (uint64_t)m, s}) { h ^= x; h *= 1099511628211ULL; }
  return h;
}

int main(int argc, char** argv) {
  int n = 128, seeds = 8;
  double rho = 0.2;
  for (int i = 1; i < argc; ++i) {
    auto nxt = [&]() { return std::string(argv[++i]); };
    std::string a = argv[i];
    if (a == "--n") n = std::stoi(nxt());
    else if (a == "--rho") rho = std::stod(nxt());
    else if (a == "--seeds") seeds = std::stoi(nxt());
  }
  PairIndex pix(n);
  const int lo = n - 1, hi = pix.numPairs();
  const int m = lo + (int)(rho * (hi - lo));
  std::printf("n,m,seed,start_lam2\n");
  for (int s = 0; s < seeds; ++s) {
    Rng rng(cellSeed(n, m, (uint64_t)s));
    const Graph g = Graph::randomPathSeeded(pix, m, rng);
    Oracle o;
    std::printf("%d,%d,%d,%.17g\n", n, m, s, o.lam2(g));
  }
  return 0;
}

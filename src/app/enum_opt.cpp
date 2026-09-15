// Exhaustive per-m optimal lambda_2 over G_{n,m}, fed by nauty's geng.
//
//   geng -cq n [mine:maxe] res/mod | enum_opt --n N --out part_R.csv [...]
//
// nauty's res/mod splitting is the parallelization: each of `mod` classes is
// an independent, disjoint stream whose union is the full canonical
// enumeration, so one enum_opt per class and a final merge gives the exact
// table. This process is a pure stream consumer: it never generates.
//
// Per graph: decode graph6, compute degrees, and PRUNE by Fiedler's bound
// lambda_2 <= delta_min for non-complete graphs (K_n, the single m = C(n,2)
// graph, is handled explicitly): if the bound cannot reach the incumbent
// best for that m minus tolerance, the eigensolve is skipped. Seeding the
// incumbents from a cheap heuristic pre-pass makes the prune strong from the
// first graph; seeds are provably safe because each seed value is attained
// by a real graph at that m, which the enumeration itself will reach.
//
// Output (atomically checkpointed): one row per m,
//   m, best_lam2 (full precision), argmax graph6, ties, candidates, solves
// then one terminal row "DONE,processed,solved,elapsed_s".
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <algorithm>

#include "Eigen.hpp"

namespace {

struct PerM {
  double best = -1.0;
  std::string argmax;
  long ties = 0;        // graphs within tol of best (argmax included)
  long candidates = 0;  // graphs seen with this m
  long solves = 0;      // eigensolves actually performed
  bool seeded = false;  // best came from the seed file, not yet witnessed
};

}  // namespace

int main(int argc, char** argv) {
  int n = 12;
  double tol = 1e-9;
  int flushSec = 60;
  std::string out, seedsPath;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if (a == "--n") n = std::stoi(nxt());
    else if (a == "--out") out = nxt();
    else if (a == "--seeds") seedsPath = nxt();
    else if (a == "--tol") tol = std::stod(nxt());
    else if (a == "--flush-sec") flushSec = std::stoi(nxt());
  }
  if (out.empty()) { std::fprintf(stderr, "enum_opt: --out required\n"); return 2; }
  const int maxM = n * (n - 1) / 2;
  std::vector<PerM> table(maxM + 1);

  if (!seedsPath.empty()) {
    FILE* f = std::fopen(seedsPath.c_str(), "r");
    if (!f) { std::fprintf(stderr, "enum_opt: cannot read seeds %s\n", seedsPath.c_str()); return 2; }
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
      int m; double v;
      if (std::sscanf(line, "%d,%lf", &m, &v) == 2 && m >= 0 && m <= maxM) {
        table[m].best = v;
        table[m].seeded = true;
      }
    }
    std::fclose(f);
  }

  auto writeOut = [&](long processed, long solved, double elapsed, bool done) {
    const std::string tmp = out + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "m,best_lam2,argmax_g6,ties,candidates,solves\n");
    for (int m = 0; m <= maxM; ++m) {
      const PerM& p = table[m];
      if (p.candidates == 0 && !p.seeded) continue;
      std::fprintf(f, "%d,%.17g,%s,%ld,%ld,%ld\n", m, p.best,
                   p.seeded && p.argmax.empty() ? "SEED" : p.argmax.c_str(),
                   p.ties, p.candidates, p.solves);
    }
    std::fprintf(f, "%s,%ld,%ld,%.1f\n", done ? "DONE" : "PARTIAL",
                 processed, solved, elapsed);
    std::fclose(f);
    std::rename(tmp.c_str(), out.c_str());
  };

  std::vector<double> L((size_t)n * n), vals;
  std::vector<int> deg(n);
  char line[128];
  long processed = 0, solved = 0;
  const auto t0 = std::chrono::steady_clock::now();
  auto lastFlush = t0;

  while (std::fgets(line, sizeof line, stdin)) {
    const size_t len = std::strlen(line);
    if (len < 2 || line[0] - 63 != n) continue;  // header byte must match n
    ++processed;

    // decode: bits run over pairs (i,j), j = 1..n-1, i = 0..j-1, 6 bits/byte,
    // most significant bit first (graph6, McKay).
    std::fill(L.begin(), L.end(), 0.0);
    std::fill(deg.begin(), deg.end(), 0);
    int m = 0, bit = 0;
    for (int j = 1; j < n; ++j) {
      for (int i = 0; i < j; ++i, ++bit) {
        const int byte = 1 + bit / 6, off = 5 - bit % 6;
        if ((line[byte] - 63) >> off & 1) {
          L[(size_t)i * n + j] = L[(size_t)j * n + i] = -1.0;
          ++deg[i]; ++deg[j]; ++m;
        }
      }
    }
    PerM& p = table[m];
    ++p.candidates;

    int dmin = n;
    for (int v = 0; v < n; ++v) dmin = std::min(dmin, deg[v]);
    // Fiedler: lambda_2 <= vertex connectivity <= delta_min unless complete.
    const double bound = (m == maxM) ? (double)n : (double)dmin;
    if (bound < p.best - tol) continue;

    for (int v = 0; v < n; ++v) L[(size_t)v * n + v] = deg[v];
    algconn::eig::symmetricValues(L, n, vals);
    ++solved; ++p.solves;
    const double l2 = vals[1];

    if (l2 > p.best + tol) {
      p.best = l2;
      p.argmax.assign(line, len - (line[len - 1] == '\n' ? 1 : 0));
      p.ties = 1;
      p.seeded = false;
    } else if (l2 >= p.best - tol) {
      ++p.ties;
      if (p.argmax.empty()) {  // first witness of a seeded value
        p.argmax.assign(line, len - (line[len - 1] == '\n' ? 1 : 0));
        p.seeded = false;
      }
    }

    if ((processed & 0xFFFF) == 0) {
      const auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration<double>(now - lastFlush).count() >= flushSec) {
        writeOut(processed, solved,
                 std::chrono::duration<double>(now - t0).count(), false);
        lastFlush = now;
      }
    }
  }
  writeOut(processed, solved,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
           true);
  return 0;
}

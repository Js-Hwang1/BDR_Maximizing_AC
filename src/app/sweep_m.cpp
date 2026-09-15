// Figure A: fixed n, sweep m across the whole feasible range, every method at
// the SAME budget B. Plotted as lambda_2 normalised by the certified bound, so
// the sparse regime's large gap is visible instead of being swamped by the
// growth of lambda_2 with m.
//
//   ./sweep_m --n 32 --stride 5 --seeds 20 --budget 20000 --threads 32 --out f.csv
//   ./sweep_m --n 12 --stride 1 --seeds 20 --budget 1000 --threads 8
//       --methods ours_local --out local.csv
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Graph.hpp"
#include "Oracle.hpp"
#include "methods/ER.hpp"
#include "methods/FV.hpp"
#include "methods/KOPT.hpp"
#include "methods/MDMD.hpp"
#include "methods/OURS.hpp"
#include "methods/RANDOM.hpp"

using namespace algconn;

static uint64_t cellSeed(int n, int m, uint64_t s) {
  uint64_t h = 1469598103934665603ULL;
  for (uint64_t x : {(uint64_t)n, (uint64_t)m, s}) { h ^= x; h *= 1099511628211ULL; }
  return h;
}
// lambda_2 <= floor(2m/n), except at K_n where lambda_2 = n exactly.
static double fiedlerBound(int n, int m) {
  return (m == n * (n - 1) / 2) ? (double)n : (double)((2 * m) / n);
}

int main(int argc, char** argv) {
  int n = 32, stride = 5, seeds = 20, threads = 32, onlyM = -1;
  long budget = 20000;
  std::string out = "-", only = "";
  bool append = false;
  int koptK = 1;   // 1-opt by default; --kopt-k 2 gives the paper's 2-opt
  for (int i = 1; i < argc; ++i) {
    auto nxt = [&]() { return std::string(argv[++i]); };
    std::string a = argv[i];
    if (a == "--n") n = std::stoi(nxt());
    else if (a == "--m") onlyM = std::stoi(nxt());
    else if (a == "--stride") stride = std::stoi(nxt());
    else if (a == "--seeds") seeds = std::stoi(nxt());
    else if (a == "--budget") budget = std::stol(nxt());
    else if (a == "--threads") threads = std::stoi(nxt());
    else if (a == "--out") out = nxt();
    else if (a == "--append") append = true;
    // Re-run one method after a change instead of redoing a sweep whose
    // other columns are unaffected.
    else if (a == "--methods") only = nxt();
    else if (a == "--kopt-k") koptK = std::stoi(nxt());
  }

  PairIndex pix(n);
  const int lo = n - 1, hi = pix.numPairs();
  struct Task { int m; uint64_t seed; };
  std::vector<Task> tasks;
  for (int m = lo; m <= hi - 1; m += stride) {
    if (onlyM >= 0 && m != onlyM) continue;
    for (int s = 0; s < seeds; ++s) tasks.push_back({m, (uint64_t)s});
  }

  const std::string koptName =
      koptK <= 1 ? "kopt" : "kopt" + std::to_string(koptK);
  auto want = [&](const std::string& name) {
    return only.empty() ||
           (std::string(",") + only + ",").find("," + name + ",") !=
               std::string::npos;
  };
  int methodCount = 0;
  for (const std::string& name :
       {std::string("random"), std::string("mdmd"), std::string("fv"),
        std::string("er"), koptName, std::string("ours")})
    methodCount += want(name);
  if (!only.empty()) {
    for (const std::string& name :
         {std::string("ours_local"), std::string("ours_separable"),
          std::string("ours_local_no_inertia"),
          std::string("ours_qhalf"), std::string("ours_qdouble")})
      methodCount += want(name);
  }

  FILE* f = (out == "-") ? stdout
                           : std::fopen(out.c_str(), append ? "a" : "w");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
  if (!append)
    std::fprintf(f, "n,m,rho,seed,method,lam2,moment_bound,fiedler_bound,"
                    "oracle_calls,real_solves,cache_hits,wall_s,mult,budget\n");
  std::fflush(f);

  std::mutex io;
  std::atomic<size_t> cursor{0}, done{0};
  auto worker = [&]() {
    std::string buf;
    buf.reserve(1 << 16);
    for (;;) {
      const size_t idx = cursor.fetch_add(1);
      if (idx >= tasks.size()) break;
      const Task t = tasks[idx];
      const double rho = (double)(t.m - lo) / (double)(hi - lo);
      Rng startRng(cellSeed(n, t.m, t.seed));
      const Graph start = Graph::randomPathSeeded(pix, t.m, startRng);
      const double fb = fiedlerBound(n, t.m);

      auto emit = [&](const Result& r, long realSolves, long cacheHits) {
        char line[512];
        std::snprintf(line, sizeof line,
                      "%d,%d,%.6f,%llu,%s,%.17g,%.17g,%.17g,%ld,%ld,%ld,%.6f,%d,%ld\n",
                      n, t.m, rho, (unsigned long long)t.seed, r.method.c_str(),
                      r.lam2, r.momentBound, fb, r.oracleCalls, realSolves,
                      cacheHits, r.wallSec, r.multiplicity, budget);
        buf += line;
      };
      auto go = [&](const Method& meth, uint64_t salt,
                    const char* reportedName = nullptr) {
        Rng rng(cellSeed(n, t.m, t.seed) ^ salt);
        Oracle o(budget);
        Result result = meth.run(start, o, rng);
        if (reportedName) result.method = reportedName;
        emit(result, o.realSolves(), o.cacheHits());
      };
      if (want("random"))        go(RANDOM(), 0xA1ULL);
      if (want("mdmd"))          go(MDMD(),           0xB2ULL);
      if (want("fv"))            go(FV(),             0xC3ULL);
      if (want("er"))            go(ER(),             0xD4ULL);
      KOPT::Config kc; kc.k = koptK;
      if (want(koptName))
        go(KOPT(kc), 0xE5ULL);
      if (want("ours"))          go(OURS(),           0xF6ULL);
      // BDR-Local is the exact production method with its depth arm removed.
      // It keeps the same start, proposal/screening rules, restart law, seed,
      // and eigensolve budget, so the comparison isolates depth rewiring.
      if (!only.empty() && want("ours_local")) {
        OURS::Config local;
        local.attack = false;
        go(OURS(local), 0xF6ULL, "ours_local");
      }
      if (!only.empty() && want("ours_separable")) {
        OURS::Config separable;
        separable.includeCoupling = false;
        go(OURS(separable), 0xF6ULL, "ours_separable");
      }
      if (!only.empty() && want("ours_local_no_inertia")) {
        OURS::Config noInertia;
        noInertia.attack = false;
        noInertia.inertiaScreen = false;
        go(OURS(noInertia), 0xF6ULL, "ours_local_no_inertia");
      }
      if (!only.empty() && want("ours_qhalf")) {
        OURS::Config qhalf;
        qhalf.proposalWidth = (n + 1) / 2;
        go(OURS(qhalf), 0xF6ULL, "ours_qhalf");
      }
      if (!only.empty() && want("ours_qdouble")) {
        OURS::Config qdouble;
        qdouble.proposalWidth = 2 * n;
        go(OURS(qdouble), 0xF6ULL, "ours_qdouble");
      }

      {
        std::lock_guard<std::mutex> lk(io);
        std::fwrite(buf.data(), 1, buf.size(), f);
        std::fflush(f);
        buf.clear();
      }
      const size_t d = ++done;
      if (d % 100 == 0) {
        std::lock_guard<std::mutex> lk(io);
        std::fprintf(stderr, "  %zu / %zu cells\n", d, tasks.size());
      }
    }
    if (!buf.empty()) { std::lock_guard<std::mutex> lk(io);
      std::fwrite(buf.data(), 1, buf.size(), f); std::fflush(f); }
  };

  std::fprintf(stderr, "n=%d  %zu cells (%zu runs)  B=%ld  %d threads\n",
               n, tasks.size(), tasks.size() * static_cast<size_t>(methodCount),
               budget, threads);
  std::vector<std::thread> pool;
  for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  if (f != stdout) std::fclose(f);
  return 0;
}

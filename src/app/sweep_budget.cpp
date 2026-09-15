// Figure B: fixed (n, rho), quality as a function of the eigensolve budget.
//
// One run per (method, seed) at B_max suffices: the Oracle records (calls,
// running best) at every improvement, so a single run yields the ENTIRE
// quality-vs-B curve. Sampling B on a grid and re-running would cost 10-100x
// more for strictly less resolution.
//
//   ./sweep_budget --n 128 --rho 0.2 --seeds 20 --budget 1000000 --threads 32
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Graph.hpp"
#include "Oracle.hpp"
#include "ParallelRestart.hpp"
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
static double fiedlerBound(int n, int m) {
  return (m == n * (n - 1) / 2) ? (double)n : (double)((2 * m) / n);
}

int main(int argc, char** argv) {
  int n = 64, seeds = 20, threads = 32, workers = 1, seed0 = 0;
  double rho = 0.2;
  long budget = 1000000;
  std::string out = "-", only = "";
  int koptK = 1;   // 1-opt by default; --kopt-k 2 gives the paper's 2-opt
  for (int i = 1; i < argc; ++i) {
    auto nxt = [&]() { return std::string(argv[++i]); };
    std::string a = argv[i];
    if (a == "--n") n = std::stoi(nxt());
    else if (a == "--rho") rho = std::stod(nxt());
    else if (a == "--seeds") seeds = std::stoi(nxt());
    else if (a == "--budget") budget = std::stol(nxt());
    else if (a == "--threads") threads = std::stoi(nxt());
    else if (a == "--workers") workers = std::stoi(nxt());
    // Seed offset, so a second process can extend an existing sweep on
    // cores that have come free instead of re-running what is done.
    else if (a == "--seed0") seed0 = std::stoi(nxt());
    else if (a == "--out") out = nxt();
    // Comma-separated method names. Re-running one method after a change
    // beats redoing a sweep whose other columns are unaffected.
    else if (a == "--methods") only = nxt();
    else if (a == "--kopt-k") koptK = std::stoi(nxt());
  }

  PairIndex pix(n);
  const int lo = n - 1, hi = pix.numPairs();
  const int m = lo + (int)(rho * (hi - lo));
  const double fb = fiedlerBound(n, m);

  const std::string koptName = koptK <= 1 ? "kopt" : "kopt" + std::to_string(koptK);
  const std::string kNames[6] = {"random", "mdmd", "fv", "er", koptName, "ours"};
  auto wanted = [&](int mi) {
    if (only.empty()) return true;
    const std::string needle = std::string(",") + kNames[mi] + ",";
    return (std::string(",") + only + ",").find(needle) != std::string::npos;
  };
  struct Task { int method; uint64_t seed; };
  std::vector<Task> tasks;
  for (int mi = 0; mi < 6; ++mi) {
    if (!wanted(mi)) continue;
    for (int s = 0; s < seeds; ++s) tasks.push_back({mi, (uint64_t)(seed0 + s)});
  }

  FILE* f = (out == "-") ? stdout : std::fopen(out.c_str(), "w");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
  // One row per improvement: the step function best(B).
  std::fprintf(f, "n,m,rho,seed,method,calls,real_solves,cache_hits,"
                  "best_lam2,fiedler_bound,final,wall_s,moment_bound\n");
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
      Rng startRng(cellSeed(n, m, t.seed));
      const Graph start = Graph::randomPathSeeded(pix, m, startRng);

      // GA omitted: not reported.
      RANDOM mRandom; MDMD mMdmd; FV mFv; ER mEr; OURS mOurs;
      KOPT::Config kc; kc.k = koptK;
      KOPT mKopt(kc);
      const Method* meths[6] = {&mRandom, &mMdmd, &mFv, &mEr, &mKopt, &mOurs};
      const uint64_t salts[6] = {0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6};

      // FV and ER restart from fresh random graphs, so their climbs are
      // i.i.d. and split across `workers` concurrent workers sharing the same
      // budget. The eigensolve count is unchanged; only wall-clock falls.
      // OURS is NOT splittable: its incumbent ratchet couples chains through
      // the global best, so K workers would each attack a weaker incumbent
      // with 1/K of the deepening -- a different, worse algorithm. It always
      // runs one worker, whatever --workers says. mdmd/random are single-shot.
      const bool restarts = (t.method == 2 || t.method == 3 || t.method == 4);
      const int w = restarts ? workers : 1;
      const ParallelResult pr = runParallelRestarts(
          *meths[t.method], start, budget, w, cellSeed(n, m, t.seed) ^ salts[t.method]);
      const Result r = pr.best;

      char line[512];
      for (auto [calls, best] : pr.trace) {
        std::snprintf(line, sizeof line,
                      "%d,%d,%.6f,%llu,%s,%ld,%ld,-1,%.17g,%.17g,0,%.6f,%.17g\n",
                      n, m, rho, (unsigned long long)t.seed, r.method.c_str(),
                      calls, calls, best, fb, r.wallSec, r.momentBound);
        buf += line;
      }
      // Terminal row, so a reader never has to infer where the curve ends.
      std::snprintf(line, sizeof line,
                    "%d,%d,%.6f,%llu,%s,%ld,%ld,%ld,%.17g,%.17g,1,%.6f,%.17g\n",
                    n, m, rho, (unsigned long long)t.seed, r.method.c_str(),
                    r.oracleCalls, pr.totalRealSolves, pr.totalCacheHits,
                    r.lam2, fb, r.wallSec, r.momentBound);
      buf += line;

      {
        std::lock_guard<std::mutex> lk(io);
        std::fwrite(buf.data(), 1, buf.size(), f);
        std::fflush(f);
        buf.clear();
      }
      const size_t d = ++done;
      { std::lock_guard<std::mutex> lk(io);
        std::fprintf(stderr, "  %zu / %zu runs (%s seed %llu)\n", d, tasks.size(),
                     r.method.c_str(), (unsigned long long)t.seed); }
    }
    if (!buf.empty()) { std::lock_guard<std::mutex> lk(io);
      std::fwrite(buf.data(), 1, buf.size(), f); std::fflush(f); }
  };

  std::fprintf(stderr, "n=%d m=%d rho=%.3f  %zu runs  B_max=%ld  %d threads  %d workers/restart-method\n",
               n, m, rho, tasks.size(), budget, threads, workers);
  std::vector<std::thread> pool;
  for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  if (f != stdout) std::fclose(f);
  return 0;
}

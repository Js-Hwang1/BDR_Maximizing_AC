#pragma once
#include <algorithm>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "Graph.hpp"
#include "Method.hpp"
#include "Oracle.hpp"
#include "Rng.hpp"

namespace algconn {

// Run a restart-based method as K independent workers sharing one budget.
//
// FV and ER climb to a leaf and then restart from a FRESH random graph, chosen
// without reference to the leaf just left. The climbs are therefore i.i.d.
// trials, and running K of them concurrently with B/K evaluations each spends
// exactly the same B as one worker running B -- the eigensolve count, which is
// what the paper reports, is unchanged. Only wall-clock differs, by up to K.
//
// This matters because parallelism in the drivers is ACROSS tasks: a sweep with
// one seed has only two long-running tasks (fv, er) and cannot occupy more than
// two cores however many are free.
//
// TAPE SEMANTICS. A worker produces a TAPE: the sequence of improvements from
// its own chain of restarts. Because restarts are i.i.d., concatenating the
// tapes -- tape0, then tape1, then tape2 -- gives exactly the sequence a single
// worker doing sequential restarts would have produced. So the merged curve is
//
//     best(b) = running max over the first b calls of tape0 ++ tape1 ++ ...
//
// and a K-worker run is directly comparable to a K=1 run. Parallelism is an
// implementation detail of wall-clock, nothing more.
//
// The earlier merge interleaved the workers by WALL-CLOCK instead, placing a
// worker's c-th call at total (c-1)*K + w + 1, as though all K ran
// simultaneously and each had spent only b/K of the shared budget. That is
// wrong: B counts eigensolves, it is not a clock. It made every parallel run
// look like it had made a quarter of the progress at small b, which showed up
// as a staircase and made K=4 sweeps incomparable with K=1 ones.
//
struct ParallelResult {
  Result best;
  std::vector<std::pair<long, double>> trace;  // merged, in TOTAL calls
  long totalCalls = 0;
  long totalRealSolves = 0;
  long totalCacheHits = 0;
  int workers = 1;
};

inline uint64_t mixSeed(uint64_t base, int worker) {
  uint64_t z = base + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(worker + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

inline ParallelResult runParallelRestarts(const Method& method, const Graph& start,
                                          long budget, int workers, uint64_t baseSeed) {
  workers = std::max(1, workers);
  const PairIndex& pix = start.pix();
  std::vector<Result> res(workers);
  std::vector<std::vector<std::pair<long, double>>> traces(workers);
  std::vector<long> spent(workers, 0);
  std::vector<long> realSolves(workers, 0);
  std::vector<long> cacheHits(workers, 0);

  auto body = [&](int w) {
    // Worker 0 keeps the shared per-seed G0 so the protocol's common start is
    // still explored; the others begin at independent random graphs, which is
    // precisely what a restart does anyway.
    Rng rng(mixSeed(baseSeed, w));
    Graph s = (w == 0) ? start : Graph::randomConnected(pix, start.m(), rng);
    const long share = budget < 0 ? -1 : budget / workers + (w < budget % workers ? 1 : 0);
    Oracle o(share);
    res[w] = method.run(s, o, rng);
    traces[w] = o.trace();
    spent[w] = o.calls();
    realSolves[w] = o.realSolves();
    cacheHits[w] = o.cacheHits();
  };

  if (workers == 1) {
    body(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (int w = 0; w < workers; ++w) pool.emplace_back(body, w);
    for (auto& t : pool) t.join();
  }

  ParallelResult out;
  out.workers = workers;
  for (int w = 0; w < workers; ++w) {
    out.totalCalls += spent[w];
    out.totalRealSolves += realSolves[w];
    out.totalCacheHits += cacheHits[w];
  }
  int bestIdx = 0;
  for (int w = 1; w < workers; ++w)
    if (res[w].lam2 > res[bestIdx].lam2) bestIdx = w;
  out.best = res[bestIdx];
  out.best.oracleCalls = out.totalCalls;

  // Concatenate the tapes in worker order: worker w's calls occupy the budget
  // interval after every earlier worker has spent its share.
  std::vector<std::pair<long, double>> pts;
  long offset = 0;
  for (int w = 0; w < workers; ++w) {
    for (auto [c, v] : traces[w]) pts.push_back({offset + c, v});
    offset += spent[w];
  }
  double running = -1.0;
  for (auto [c, v] : pts) {
    if (v <= running) continue;
    running = v;
    // Two workers can improve on the same local step, which maps to the same
    // total-call count. Keep one point per abscissa so the curve stays a
    // proper step function rather than a multivalued scatter.
    if (!out.trace.empty() && out.trace.back().first == c)
      out.trace.back().second = running;
    else
      out.trace.push_back({c, running});
  }
  return out;
}

}  // namespace algconn

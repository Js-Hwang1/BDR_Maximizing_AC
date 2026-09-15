#include "Oracle.hpp"

#include <stdexcept>

#include "Eigen.hpp"

namespace algconn {

Oracle::Eval& Oracle::compute(const Graph& g) {
  ++queries_;
  const std::vector<uint64_t>& key = g.edges().words();
  if (hasLastFull_ && key == lastFullKey_) {
    ++cacheHits_;
    return lastFullEval_;
  }
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    ++cacheHits_;
    return it->second;
  }

  Eval ev;
  const int n = g.n();
  if (!g.isConnected()) {
    ev.lam2 = 0.0;
    ev.lam3 = 0.0;
    ev.multiplicity = 0;
    ev.v2.assign(n, 0.0);
    ev.v3.assign(n, 0.0);
    return cache_.emplace(key, std::move(ev)).first->second;
  }

  if (exhausted()) throw std::runtime_error("eigensolve budget exhausted");
  std::vector<double> a = g.laplacian(), vals;
  eig::symmetric(a, n, vals, true);
  ++solvesVectors_;
  ++calls_;
  ev.lam2 = vals[1];
  ev.lam3 = n >= 3 ? vals[2] : vals[1];
  ev.multiplicity = eig::lambda2Multiplicity(vals);
  ev.v2.resize(n);
  ev.v3.resize(n);
  for (int r = 0; r < n; ++r) {
    ev.v2[r] = a[static_cast<size_t>(r) * n + 1];
    ev.v3[r] = n >= 3 ? a[static_cast<size_t>(r) * n + 2] : ev.v2[r];
  }
  if (refM_ < 0) refM_ = g.m();
  else if (g.m() != refM_) ++offManifold_;
  // Only a FEASIBLE candidate may set the running best. The GA eigensolves the
  // UNION of two parents, which carries more than m edges and therefore, by
  // monotonicity of lambda_2 under edge addition, a larger value than anything
  // in G_{n,m}. Letting such a call raise best_ would inflate the
  // quality-vs-budget trace that is the paper's central figure.
  if (g.m() == refM_ && ev.lam2 > best_) {
    best_ = ev.lam2;
    trace_.push_back({calls_, best_});
  }
  return cache_.emplace(key, std::move(ev)).first->second;
}

void Oracle::eigensystem(const Graph& g, std::vector<double>& vals,
                         std::vector<double>& vecs) {
  if (!g.isConnected())
    throw std::runtime_error("eigensystem of a disconnected graph");
  ++queries_;
  if (exhausted()) throw std::runtime_error("eigensolve budget exhausted");
  const int n = g.n();
  vecs = g.laplacian();
  eig::symmetric(vecs, n, vals, true);
  ++solvesVectors_;
  ++calls_;

  Eval ev;
  ev.lam2 = vals[1];
  ev.lam3 = n >= 3 ? vals[2] : vals[1];
  ev.multiplicity = eig::lambda2Multiplicity(vals);
  ev.v2.resize(n);
  ev.v3.resize(n);
  for (int r = 0; r < n; ++r) {
    ev.v2[r] = vecs[static_cast<size_t>(r) * n + 1];
    ev.v3[r] = n >= 3 ? vecs[static_cast<size_t>(r) * n + 2] : ev.v2[r];
  }
  const std::vector<uint64_t>& key = g.edges().words();
  lastFullKey_ = key;
  lastFullEval_ = std::move(ev);
  hasLastFull_ = true;
  if (refM_ < 0) refM_ = g.m();
  else if (g.m() != refM_) ++offManifold_;
  if (g.m() == refM_ && vals[1] > best_) {
    best_ = vals[1];
    trace_.push_back({calls_, best_});
  }
}

double Oracle::lam2(const Graph& g) { return compute(g).lam2; }
const Oracle::Eval& Oracle::eigenpair(const Graph& g) { return compute(g); }
int Oracle::multiplicity(const Graph& g) { return compute(g).multiplicity; }
double Oracle::lam3(const Graph& g) { return compute(g).lam3; }

}  // namespace algconn

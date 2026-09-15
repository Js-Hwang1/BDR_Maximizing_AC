#include "Graph.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace algconn {
namespace {

// Adjacency lists carrying the pair index of each incident edge, so a DFS can
// skip the edge it arrived on by identity rather than by neighbour vertex.
struct Adj {
  std::vector<std::vector<std::pair<int, int>>> nb;  // (neighbour, pairIndex)
};

Adj buildAdj(const Graph& g) {
  Adj a;
  a.nb.assign(g.n(), {});
  for (int k : g.edges().ones()) {
    const int u = g.pix().i(k), v = g.pix().j(k);
    a.nb[u].push_back({v, k});
    a.nb[v].push_back({u, k});
  }
  return a;
}

}  // namespace

Graph::Graph(const PairIndex& pix, Bitset edges)
    : pix_(&pix), e_(std::move(edges)) {
  m_ = e_.count();
  rebuildDegrees();
}

void Graph::rebuildDegrees() {
  deg_.assign(pix_->n(), 0);
  for (int k : e_.ones()) { ++deg_[pix_->i(k)]; ++deg_[pix_->j(k)]; }
  sumDegSq_ = 0;
  for (int d : deg_) sumDegSq_ += static_cast<long>(d) * d;
}

Graph Graph::randomConnected(const PairIndex& pix, int m, Rng& rng) {
  const int n = pix.n();
  const int np = pix.numPairs();
  if (m < n - 1 || m > np)
    throw std::invalid_argument("m outside [n-1, binom(n,2)]");

  Bitset e(np);

  // RANDOM spanning tree: attach each vertex of a random permutation to a
  // uniformly chosen earlier one. Exactly n-1 edges, connected by construction.
  std::vector<int> perm(n);
  for (int k = 0; k < n; ++k) perm[k] = k;
  for (int k = n - 1; k > 0; --k) std::swap(perm[k], perm[rng.below(k + 1)]);
  for (int k = 1; k < n; ++k)
    e.set(pix.index(perm[k], perm[rng.below(k)]));
  if (e.count() != n - 1) throw std::logic_error("spanning tree edge count");

  // Add exactly m-(n-1) further pairs, sampled without replacement.
  int extra = m - (n - 1);
  if (extra > 0) {
    std::vector<int> free;
    free.reserve(np - (n - 1));
    for (int k = 0; k < np; ++k)
      if (!e.test(k)) free.push_back(k);
    for (int t = 0; t < extra; ++t) {
      const int pick = static_cast<int>(rng.below(
          static_cast<uint32_t>(free.size() - t))) + t;
      std::swap(free[t], free[pick]);
      e.set(free[t]);
    }
  }

  Graph g(pix, std::move(e));
  if (g.m() != m) throw std::logic_error("randomConnected produced wrong m");
  return g;
}

Graph Graph::randomPathSeeded(const PairIndex& pix, int m, Rng& rng) {
  const int n = pix.n();
  const int np = pix.numPairs();
  if (m < n - 1 || m > np)
    throw std::invalid_argument("m outside [n-1, binom(n,2)]");

  Bitset e(np);
  std::vector<int> perm(n);
  for (int k = 0; k < n; ++k) perm[k] = k;
  for (int k = n - 1; k > 0; --k) std::swap(perm[k], perm[rng.below(k + 1)]);
  for (int k = 0; k + 1 < n; ++k) e.set(pix.index(perm[k], perm[k + 1]));
  if (e.count() != n - 1) throw std::logic_error("path seed edge count");

  const int extra = m - (n - 1);
  if (extra > 0) {
    std::vector<int> free;
    free.reserve(np - (n - 1));
    for (int k = 0; k < np; ++k)
      if (!e.test(k)) free.push_back(k);
    for (int t = 0; t < extra; ++t) {
      const int pick =
          static_cast<int>(rng.below(static_cast<uint32_t>(free.size() - t))) + t;
      std::swap(free[t], free[pick]);
      e.set(free[t]);
    }
  }
  Graph g(pix, std::move(e));
  if (g.m() != m) throw std::logic_error("randomPathSeeded produced wrong m");
  return g;
}

void Graph::applySwap(Swap s) {
  if (s.add < 0 || s.remove < 0) throw std::invalid_argument("uninitialised swap");
  if (e_.test(s.add)) throw std::invalid_argument("swap.add is already an edge");
  if (!e_.test(s.remove)) throw std::invalid_argument("swap.remove is not an edge");

  const int ai = pix_->i(s.add), aj = pix_->j(s.add);
  const int ri = pix_->i(s.remove), rj = pix_->j(s.remove);

  // Maintain sum of squared degrees incrementally: (d+1)^2 - d^2 = 2d+1 and
  // (d-1)^2 - d^2 = -2d+1. Order matters when endpoints coincide.
  sumDegSq_ += 2L * deg_[ai] + 1; ++deg_[ai];
  sumDegSq_ += 2L * deg_[aj] + 1; ++deg_[aj];
  sumDegSq_ += -2L * deg_[ri] + 1; --deg_[ri];
  sumDegSq_ += -2L * deg_[rj] + 1; --deg_[rj];

  e_.set(s.add);
  e_.clear(s.remove);
  // m_ is unchanged by construction: one set, one clear.
}

bool Graph::isConnected() const {
  const int n = pix_->n();
  const Adj a = buildAdj(*this);
  std::vector<char> seen(n, 0);
  std::vector<int> stack{0};
  seen[0] = 1;
  int reached = 1;
  while (!stack.empty()) {
    const int v = stack.back();
    stack.pop_back();
    for (auto [w, k] : a.nb[v]) {
      (void)k;
      if (!seen[w]) { seen[w] = 1; ++reached; stack.push_back(w); }
    }
  }
  return reached == n;
}

std::vector<int> Graph::bridgePairs() const {
  const int n = pix_->n();
  const Adj a = buildAdj(*this);
  std::vector<int> disc(n, -1), low(n, 0);
  std::vector<int> out;
  int timer = 0;

  struct Frame { int v; int inEdge; size_t ptr; };
  for (int root = 0; root < n; ++root) {
    if (disc[root] != -1) continue;
    disc[root] = low[root] = timer++;
    std::vector<Frame> st{{root, -1, 0}};
    while (!st.empty()) {
      Frame& f = st.back();
      if (f.ptr < a.nb[f.v].size()) {
        auto [w, k] = a.nb[f.v][f.ptr++];
        if (k == f.inEdge) continue;  // do not walk back along the arrival edge
        if (disc[w] == -1) {
          disc[w] = low[w] = timer++;
          st.push_back({w, k, 0});
        } else {
          low[f.v] = std::min(low[f.v], disc[w]);
        }
      } else {
        const int v = f.v;
        const int inEdge = f.inEdge;
        st.pop_back();
        if (!st.empty()) {
          const int p = st.back().v;
          low[p] = std::min(low[p], low[v]);
          if (low[v] > disc[p]) out.push_back(inEdge);
        }
      }
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

bool Graph::swapKeepsConnected(Swap s) const {
  Graph t = *this;
  t.applySwap(s);
  return t.isConnected();
}

SwapConnectivity::SwapConnectivity(const Graph& graph)
    : pix_(&graph.pix()),
      entry_(graph.n(), -1),
      exit_(graph.n(), -1),
      bridgeChild_(graph.numPairs(), -1) {
  const Adj adjacency = buildAdj(graph);
  const int n = graph.n();
  std::vector<int> low(n, -1), parent(n, -1), parentEdge(n, -1);
  struct Frame {
    int vertex;
    size_t next;
  };

  int timer = 0;
  entry_[0] = low[0] = timer++;
  std::vector<Frame> stack{{0, 0}};
  while (!stack.empty()) {
    Frame& frame = stack.back();
    const int vertex = frame.vertex;
    if (frame.next < adjacency.nb[vertex].size()) {
      const auto [neighbor, edge] = adjacency.nb[vertex][frame.next++];
      if (edge == parentEdge[vertex]) continue;
      if (entry_[neighbor] < 0) {
        parent[neighbor] = vertex;
        parentEdge[neighbor] = edge;
        entry_[neighbor] = low[neighbor] = timer++;
        stack.push_back({neighbor, 0});
      } else {
        low[vertex] = std::min(low[vertex], entry_[neighbor]);
      }
      continue;
    }

    exit_[vertex] = timer - 1;
    stack.pop_back();
    if (parent[vertex] < 0) continue;
    const int p = parent[vertex];
    low[p] = std::min(low[p], low[vertex]);
    if (low[vertex] > entry_[p])
      bridgeChild_[parentEdge[vertex]] = vertex;
  }
  if (timer != n)
    throw std::invalid_argument("swap connectivity requires a connected graph");
}

bool SwapConnectivity::keepsConnected(Swap swap) const {
  const int child = bridgeChild_[swap.remove];
  if (child < 0) return true;
  auto inBridgeSubtree = [&](int vertex) {
    return entry_[child] <= entry_[vertex] && entry_[vertex] <= exit_[child];
  };
  return inBridgeSubtree(pix_->i(swap.add)) !=
         inBridgeSubtree(pix_->j(swap.add));
}

std::vector<double> Graph::laplacian() const {
  const int n = pix_->n();
  std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
  for (int k : e_.ones()) {
    const int u = pix_->i(k), v = pix_->j(k);
    L[static_cast<size_t>(u) * n + v] = -1.0;
    L[static_cast<size_t>(v) * n + u] = -1.0;
  }
  for (int v = 0; v < n; ++v) L[static_cast<size_t>(v) * n + v] = deg_[v];
  return L;
}

double Graph::momentUpperBound() const {
  const int n = pix_->n();
  if (n < 3) return static_cast<double>(n);
  const double k = n - 1;
  const double twoM = 2.0 * m_;
  const double mu = twoM / k;
  const double second = (static_cast<double>(sumDegSq_) + twoM) / k;
  const double var = std::max(second - mu * mu, 0.0);
  return mu - std::sqrt(var) / std::sqrt(k - 1.0);
}

std::vector<std::pair<int, int>> Graph::edgeList() const {
  std::vector<std::pair<int, int>> out;
  out.reserve(m_);
  for (int k : e_.ones()) out.push_back({pix_->i(k), pix_->j(k)});
  return out;
}

void Graph::checkFeasible() const {
  if (e_.count() != m_)
    throw std::logic_error("edge count drifted from m");
  long chk = 0;
  std::vector<int> d(pix_->n(), 0);
  for (int k : e_.ones()) { ++d[pix_->i(k)]; ++d[pix_->j(k)]; }
  for (int v = 0; v < pix_->n(); ++v) {
    if (d[v] != deg_[v]) throw std::logic_error("degree cache is stale");
    chk += static_cast<long>(d[v]) * d[v];
  }
  if (chk != sumDegSq_) throw std::logic_error("sumDegSquared cache is stale");
  if (!isConnected()) throw std::logic_error("graph is disconnected");
}

}  // namespace algconn

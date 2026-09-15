#include "methods/GA.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace algconn {
namespace {

// Component id of every vertex after deleting pair `skip`.
std::vector<int> componentsWithout(const Graph& g, int skip) {
  const int n = g.n();
  std::vector<std::vector<int>> nb(n);
  for (int k : g.edges().ones()) {
    if (k == skip) continue;
    nb[g.pix().i(k)].push_back(g.pix().j(k));
    nb[g.pix().j(k)].push_back(g.pix().i(k));
  }
  std::vector<int> comp(n, -1);
  int c = 0;
  for (int s = 0; s < n; ++s) {
    if (comp[s] != -1) continue;
    std::vector<int> st{s};
    comp[s] = c;
    while (!st.empty()) {
      const int v = st.back();
      st.pop_back();
      for (int w : nb[v])
        if (comp[w] == -1) { comp[w] = c; st.push_back(w); }
    }
    ++c;
  }
  return comp;
}

// Sample an index proportional to exp(score/tau), numerically stabilised.
int softmaxSample(const std::vector<double>& score, double tau, Rng& rng) {
  const size_t k = score.size();
  double mx = -1e300;
  for (double s : score) mx = std::max(mx, s / tau);
  std::vector<double> p(k);
  double sum = 0.0;
  for (size_t i = 0; i < k; ++i) { p[i] = std::exp(score[i] / tau - mx); sum += p[i]; }
  double r = rng.uniform() * sum, c = 0.0;
  for (size_t i = 0; i < k; ++i) { c += p[i]; if (r <= c) return static_cast<int>(i); }
  return static_cast<int>(k) - 1;
}

}  // namespace

Graph GA::crossoverUnionPrune(const Graph& pa, const Graph& pb, Oracle& oracle,
                              Rng& rng) {
  (void)rng;
  const PairIndex& pix = pa.pix();
  const int target = pa.m();

  // 1. Union of the parents. This has MORE than m edges in general.
  Bitset u(pix.numPairs());
  for (int k : pa.edges().ones()) u.set(k);
  for (int k : pb.edges().ones()) u.set(k);
  Graph child(pix, std::move(u));
  if (child.m() <= target) return child;

  // 2. The Fiedler vector OF THE UNION -- the point of the design. It reflects
  //    the merged connectivity pattern, not either parent's.
  const std::vector<double> v2 = oracle.eigenpair(child).v2;

  // 3. Score every union edge by its Fiedler gap, ascending (lowest first =
  //    most expendable).
  std::vector<std::pair<double, int>> score;
  score.reserve(child.m());
  for (int k : child.edges().ones()) {
    const double d = v2[pix.i(k)] - v2[pix.j(k)];
    score.push_back({d * d, k});
  }
  std::sort(score.begin(), score.end(),
            [](const auto& a, const auto& b) {
              return a.first != b.first ? a.first < b.first : a.second < b.second;
            });

  // 4. Prune to exactly m edges, keeping the highest-gap structure.
  //
  //    Deleting lowest-gap edges one at a time, skipping any whose removal
  //    would disconnect, is precisely REVERSE-DELETE: run to completion it
  //    yields the MAXIMUM spanning tree with respect to the gap. Stopping after
  //    (union_m - m) deletions therefore keeps exactly
  //        max-spanning-tree  U  the highest-gap remaining edges.
  //    Computing it that way needs one Kruskal pass with union-find instead of
  //    a bridge recomputation per deletion, and is the same set (verified
  //    against the naive loop in the tests).
  std::vector<int> parent(pix.n());
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };

  Bitset keep(pix.numPairs());
  int kept = 0;
  // Descending gap: Kruskal for the maximum spanning tree.
  for (auto it = score.rbegin(); it != score.rend(); ++it) {
    const int k = it->second;
    const int ra = find(pix.i(k)), rb = find(pix.j(k));
    if (ra != rb) { parent[ra] = rb; keep.set(k); ++kept; }
  }
  // Then the highest-gap non-tree edges until m.
  for (auto it = score.rbegin(); it != score.rend() && kept < target; ++it) {
    const int k = it->second;
    if (!keep.test(k)) { keep.set(k); ++kept; }
  }
  child = Graph(pix, std::move(keep));

  if (child.m() != target)
    throw std::logic_error("crossover failed to prune the union to m edges");
  return child;
}

int GA::rewireToward(Graph& g, const Graph& target, int k, Rng& rng) {
  std::vector<int> out = g.edges().minus(target.edges());
  std::vector<int> in = target.edges().minus(g.edges());
  if (out.size() != in.size())
    throw std::logic_error("parents differ in edge count");
  const int d = static_cast<int>(out.size());
  if (d == 0 || k <= 0) return 0;
  k = std::min(k, d);

  for (int t = static_cast<int>(out.size()) - 1; t > 0; --t)
    std::swap(out[t], out[rng.below(t + 1)]);
  for (int t = static_cast<int>(in.size()) - 1; t > 0; --t)
    std::swap(in[t], in[rng.below(t + 1)]);

  int applied = 0;
  std::vector<char> usedIn(in.size(), 0);
  for (int oi = 0; oi < d && applied < k; ++oi) {
    const int rem = out[oi];
    if (!g.has(rem)) continue;
    std::vector<char> isBridge(g.numPairs(), 0);
    for (int b : g.bridgePairs()) isBridge[b] = 1;
    int chosen = -1;
    if (!isBridge[rem]) {
      for (size_t ii = 0; ii < in.size(); ++ii)
        if (!usedIn[ii] && !g.has(in[ii])) { chosen = static_cast<int>(ii); break; }
    } else {
      const std::vector<int> comp = componentsWithout(g, rem);
      for (size_t ii = 0; ii < in.size(); ++ii) {
        if (usedIn[ii] || g.has(in[ii])) continue;
        if (comp[g.pix().i(in[ii])] != comp[g.pix().j(in[ii])]) {
          chosen = static_cast<int>(ii);
          break;
        }
      }
    }
    if (chosen < 0) continue;
    g.applySwap(Swap{in[chosen], rem});
    usedIn[chosen] = 1;
    ++applied;
  }
  return applied;
}

Result GA::run(Graph start, Oracle& oracle, Rng& rng) const {
  const auto t0 = std::chrono::steady_clock::now();
  const PairIndex& pix = start.pix();
  const int n = start.n();
  const int m = start.m();

  // Reference scaling: pop = n, generations = 3n, mutations per child = n/2.
  const int P = cfg_.popQuad > 0 ? std::max(2, (int)(cfg_.popQuad * n * n))
               : cfg_.popMult > 0 ? std::max(2, (int)(cfg_.popMult * n))
                                  : cfg_.popSize;
  const int G = cfg_.genQuad > 0 ? std::max(1, (int)(cfg_.genQuad * n * n))
               : cfg_.genMult > 0 ? std::max(1, (int)(cfg_.genMult * n))
                                  : cfg_.generations;
  const int numMut = cfg_.mutMult > 0 ? std::max(1, (int)(cfg_.mutMult * n))
                                      : cfg_.numMutations;
  const int refreshEvery =
      cfg_.refreshEvery > 0 ? cfg_.refreshEvery : std::max(2, n / 4);
  int nElite = (int)(cfg_.eliteFrac * P);
  if (nElite < 1) nElite = 1;
  if (nElite >= P) nElite = P - 1;

  bool stop = false;
  Graph bestG = start;
  double bestL = -1.0;

  auto evaluate = [&](const Graph& g) -> double {
    const double v = oracle.lam2(g);
    if (v > bestL) { bestL = v; bestG = g; }
    return v;
  };

  // --- FV-guided softmax mutation ------------------------------------------
  auto mutate = [&](Graph& g) {
    std::vector<double> v2;
    std::vector<char> isBridge(pix.numPairs(), 0);
    for (int s = 0; s < numMut; ++s) {
      if (oracle.exhausted()) { stop = true; return; }
      if (s % refreshEvery == 0 && !cfg_.randomMutation) {
        try { v2 = oracle.eigenpair(g).v2; }
        catch (const std::exception&) { stop = true; return; }
      }
      std::fill(isBridge.begin(), isBridge.end(), 0);
      for (int b : g.bridgePairs()) isBridge[b] = 1;

      std::vector<int> addIdx, remIdx;
      for (int k = 0; k < pix.numPairs(); ++k) {
        if (g.has(k)) { if (!isBridge[k]) remIdx.push_back(k); }
        else addIdx.push_back(k);
      }
      if (addIdx.empty() || remIdx.empty()) return;

      auto gap = [&](int k) {
        if (cfg_.randomMutation || v2.empty()) return 1.0;
        const double d = v2[pix.i(k)] - v2[pix.j(k)];
        return d * d;
      };
      const double tau = cfg_.randomMutation ? 1.0 : cfg_.tau;

      // ADD: prefer HIGH Fiedler gap.
      std::vector<double> sa(addIdx.size());
      for (size_t i = 0; i < addIdx.size(); ++i) sa[i] = gap(addIdx[i]);
      const int a = addIdx[softmaxSample(sa, tau, rng)];

      // REMOVE: prefer LOW gap, via the reference's inversion maxGap - gap.
      std::vector<double> sr(remIdx.size());
      double mx = 0.0;
      for (size_t i = 0; i < remIdx.size(); ++i) {
        sr[i] = gap(remIdx[i]);
        mx = std::max(mx, sr[i]);
      }
      if (!cfg_.randomMutation)
        for (double& x : sr) x = mx - x;
      const int r = remIdx[softmaxSample(sr, tau, rng)];

      // Add first, then remove -- matching the reference. `r` was a non-bridge
      // before the insertion, and inserting an edge can only destroy bridges,
      // so the swap is connectivity-preserving.
      g.applySwap(Swap{a, r});
    }
  };

  // --- population ----------------------------------------------------------
  std::vector<Graph> pop;
  pop.reserve(P);
  pop.push_back(start);
  for (int i = 1; i < P; ++i) pop.push_back(Graph::randomPathSeeded(pix, m, rng));

  std::vector<double> fit(P, -1.0);
  for (int i = 0; i < P && !stop; ++i) {
    if (oracle.exhausted()) { stop = true; break; }
    fit[i] = evaluate(pop[i]);
  }

  int generations = 0;
  long lastCalls = oracle.calls();
  int stagnant = 0;
  std::vector<int> order(P);

  for (int gen = 0; gen < G && !stop; ++gen) {
    if (oracle.calls() == lastCalls) {
      if (++stagnant >= cfg_.stagnationLimit) break;
    } else { stagnant = 0; lastCalls = oracle.calls(); }

    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return fit[a] > fit[b]; });

    std::vector<Graph> next;
    next.reserve(P);
    for (int e = 0; e < nElite; ++e) next.push_back(pop[order[e]]);

    auto tournamentPick = [&]() -> const Graph& {
      int best = (int)rng.below((uint32_t)P);
      for (int t = 1; t < cfg_.tournament && t < P; ++t) {
        const int c = (int)rng.below((uint32_t)P);
        if (fit[c] > fit[best]) best = c;
      }
      return pop[best];
    };

    while ((int)next.size() < P && !stop) {
      const Graph& p1 = tournamentPick();
      const Graph& p2 = tournamentPick();
      Graph child = p1;
      try {
        child = crossoverUnionPrune(p1, p2, oracle, rng);
      } catch (const std::exception&) { stop = true; break; }
      if (rng.uniform() < cfg_.mutationRate) mutate(child);
      next.push_back(std::move(child));
    }
    if (next.empty()) break;

    std::vector<double> nextFit(next.size(), -1.0);
    for (size_t i = 0; i < next.size() && !stop; ++i) {
      if (oracle.exhausted()) { stop = true; break; }
      nextFit[i] = evaluate(next[i]);
    }
    // Keep the population at full size even if the budget cut generation short.
    while ((int)next.size() < P) { next.push_back(pop[order[next.size()]]);
                                   nextFit.push_back(fit[order[nextFit.size()]]); }
    pop.swap(next);
    fit.swap(nextFit);
    ++generations;
  }

  Graph bg = bestG;
  bg.checkFeasible();
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return Result{name(),        bg.n(), bg.m(), bestL,
                bg.edgeList(), oracle.calls(), oracle.queries(),
                wall,          0,      generations, bg.momentUpperBound(),
                oracle.multiplicity(bg)};
}

}  // namespace algconn

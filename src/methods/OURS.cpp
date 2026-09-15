#include "methods/OURS.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Eigen.hpp"
#include "methods/detail/GroundedInverse.hpp"

namespace algconn {
namespace {

struct ScoredSwap {
  double score;
  Swap swap;
};

struct CoupledDomain {
  std::vector<double> resistance;
  std::vector<int> additions;
  std::vector<int> removals;
  size_t rank = 0;
};

std::vector<Swap> coupledProposalsImpl(
    const Graph& g, const detail::GroundedInverse& inverse,
    const std::vector<char>* allowedAdds,
    const std::vector<char>* allowedRemovals, bool responsesOnly,
    int proposalWidth, bool includeCoupling,
    size_t* distinctPairsExamined = nullptr);

CoupledDomain buildCoupledDomain(
    const Graph& g, const detail::GroundedInverse& inverse,
    const std::vector<char>* allowedAdds,
    const std::vector<char>* allowedRemovals, int proposalWidth) {
  const PairIndex& pix = g.pix();
  CoupledDomain domain;
  domain.resistance.assign(pix.numPairs(), 0.0);
  domain.additions.reserve(pix.numPairs() - g.m());
  domain.removals.reserve(g.m());
  for (int pair = 0; pair < pix.numPairs(); ++pair) {
    const bool removal = g.has(pair);
    const bool eligible = removal
        ? (!allowedRemovals || (*allowedRemovals)[pair])
        : (!allowedAdds || (*allowedAdds)[pair]);
    if (!eligible) continue;
    domain.resistance[pair] =
        inverse.effectiveResistance(pix.i(pair), pix.j(pair));
    (removal ? domain.removals : domain.additions).push_back(pair);
  }
  std::sort(domain.additions.begin(), domain.additions.end(),
            [&](int left, int right) {
    if (domain.resistance[left] != domain.resistance[right])
      return domain.resistance[left] > domain.resistance[right];
    return left < right;
  });
  std::sort(domain.removals.begin(), domain.removals.end(),
            [&](int left, int right) {
    if (domain.resistance[left] != domain.resistance[right])
      return domain.resistance[left] < domain.resistance[right];
    return left < right;
  });
  const int width = proposalWidth > 0 ? proposalWidth : g.n() - 1;
  domain.rank = static_cast<size_t>(std::max(1, width));
  return domain;
}

double proposalScore(const Graph& g, const detail::GroundedInverse& inverse,
                     const CoupledDomain& domain, int add, int remove,
                     bool includeCoupling) {
  const double separable =
      domain.resistance[add] - domain.resistance[remove];
  if (!includeCoupling) return separable;
  const PairIndex& pix = g.pix();
  const double transfer = inverse.transfer(
      pix.i(add), pix.j(add), pix.i(remove), pix.j(remove));
  return separable -
         domain.resistance[add] * domain.resistance[remove] +
         transfer * transfer;
}

template <class Visitor>
void visitRankMatchedUnion(const Graph& g,
                           const detail::GroundedInverse& inverse,
                           const CoupledDomain& domain, bool includeCoupling,
                           Visitor&& visitor,
                           size_t* examined = nullptr) {
  const size_t addRows = std::min(domain.rank, domain.additions.size());
  const size_t removalColumns =
      std::min(domain.rank, domain.removals.size());
  // Traverse the union itself, not the full A-by-R Cartesian product with a
  // skip inside. The latter executes O(n^4) loop iterations at fixed density
  // even though only O(n^3) pairs are counted. These two disjoint fibers
  // execute exactly qR + (A-q)q iterations.
  for (size_t ai = 0; ai < addRows; ++ai) {
    for (size_t ri = 0; ri < domain.removals.size(); ++ri) {
      if (examined) ++*examined;
      const int add = domain.additions[ai];
      const int remove = domain.removals[ri];
      visitor(ai, ri, add, remove,
              proposalScore(g, inverse, domain, add, remove,
                            includeCoupling));
    }
  }
  for (size_t ai = addRows; ai < domain.additions.size(); ++ai) {
    for (size_t ri = 0; ri < removalColumns; ++ri) {
      if (examined) ++*examined;
      const int add = domain.additions[ai];
      const int remove = domain.removals[ri];
      visitor(ai, ri, add, remove,
              proposalScore(g, inverse, domain, add, remove,
                            includeCoupling));
    }
  }
}

template <class Key>
void stableRadixField(std::vector<ScoredSwap>& values, int bytes, Key key) {
  std::vector<ScoredSwap> scratch(values.size());
  for (int byte = 0; byte < bytes; ++byte) {
    std::array<size_t, 256> count{};
    const int shift = 8 * byte;
    for (const ScoredSwap& value : values)
      ++count[(key(value) >> shift) & 0xffU];
    size_t prefix = 0;
    for (size_t& bucket : count) {
      const size_t frequency = bucket;
      bucket = prefix;
      prefix += frequency;
    }
    for (const ScoredSwap& value : values)
      scratch[count[(key(value) >> shift) & 0xffU]++] = value;
    values.swap(scratch);
  }
}

void radixSortScored(std::vector<ScoredSwap>& values) {
  // Stable least-significant-field passes implement the exact order
  // (score descending, addition ascending, removal ascending) in a fixed
  // number of word operations per proposal. Canonicalize signed zero because
  // the comparison order treats +0 and -0 as the same score.
  stableRadixField(values, sizeof(uint32_t), [](const ScoredSwap& value) {
    return static_cast<uint64_t>(static_cast<uint32_t>(value.swap.remove));
  });
  stableRadixField(values, sizeof(uint32_t), [](const ScoredSwap& value) {
    return static_cast<uint64_t>(static_cast<uint32_t>(value.swap.add));
  });
  stableRadixField(values, sizeof(uint64_t), [](const ScoredSwap& value) {
    const double score = value.score == 0.0 ? 0.0 : value.score;
    const uint64_t bits = std::bit_cast<uint64_t>(score);
    const uint64_t ascending =
        (bits >> 63) != 0 ? ~bits : (bits ^ (uint64_t{1} << 63));
    return ~ascending;
  });
}

// A-posteriori eigenvalue-error envelope for the computed eigensystem. Let
// E=LV-VD, H=I-VV^T, and let Q be the orthogonal polar factor of V. Then
//
//   L-VDV^T = EV^T + LH.
//
// If h=||I-V^T V||_F<1, polar-decomposition bounds control the additional
// distance from VDV^T to QDQ^T, whose eigenvalues are exactly diag(D). Weyl's
// theorem then gives the envelope below. E is formed by sparse Laplacian
// multiplication in O(mn), while the measured orthogonality defect costs
// O(n^3). No fitted tolerance or unproved near-orthogonality assumption is
// used.
double eigensystemError(const Graph& g, const std::vector<double>& vals,
                        const std::vector<double>& vecs) {
  const int n = g.n();
  std::vector<double> residual(static_cast<size_t>(n) * n, 0.0);
  for (int row = 0; row < n; ++row)
    for (int mode = 0; mode < n; ++mode)
      residual[static_cast<size_t>(row) * n + mode] =
          (static_cast<double>(g.degree(row)) - vals[mode]) *
          vecs[static_cast<size_t>(row) * n + mode];
  for (int pair : g.edges().ones()) {
    const int u = g.pix().i(pair);
    const int v = g.pix().j(pair);
    for (int mode = 0; mode < n; ++mode) {
      residual[static_cast<size_t>(u) * n + mode] -=
          vecs[static_cast<size_t>(v) * n + mode];
      residual[static_cast<size_t>(v) * n + mode] -=
          vecs[static_cast<size_t>(u) * n + mode];
    }
  }
  double residual2 = 0.0;
  for (double entry : residual) residual2 += entry * entry;

  double orthogonality2 = 0.0;
  double gram2 = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double gram = 0.0;
      for (int k = 0; k < n; ++k)
        gram += vecs[static_cast<size_t>(k) * n + i] *
                vecs[static_cast<size_t>(k) * n + j];
      gram2 += gram * gram;
      if (i == j) gram -= 1.0;
      orthogonality2 += gram * gram;
    }
  }
  if (!std::isfinite(residual2) || !std::isfinite(gram2) ||
      !std::isfinite(orthogonality2))
    throw std::runtime_error("computed eigensystem contains nonfinite data");
  const int maximumDegree =
      *std::max_element(g.degrees().begin(), g.degrees().end());
  const double vectorNormBound = std::sqrt(std::sqrt(gram2));
  const double orthogonality = std::sqrt(orthogonality2);
  if (!(orthogonality < 1.0))
    throw std::runtime_error(
        "computed eigensystem is too nonorthogonal to certify");
  double spectralScale = 0.0;
  for (double value : vals) {
    if (!std::isfinite(value))
      throw std::runtime_error("computed eigensystem contains nonfinite data");
    spectralScale = std::max(spectralScale, std::fabs(value));
  }
  const double polarMultiplier =
      (std::sqrt(1.0 + orthogonality) + 1.0) /
      (std::sqrt(1.0 - orthogonality) + 1.0);
  return std::sqrt(residual2) * vectorNormBound +
         (2.0 * maximumDegree + spectralScale * polarMultiplier) *
             orthogonality;
}

double numericalResolution(double value, double error) {
  // The secular sums contain reciprocals of (lambda_i - threshold). Evaluating
  // within roundoff of an anchor pole makes the 2-by-2 determinant lose all
  // sign information. The classical square-root balance between data error
  // and reciprocal amplification supplies the pole separation; its only
  // inputs are machine epsilon and the measured spectral scale.
  const double scale = std::max(1.0, std::fabs(value));
  const double poleSeparation =
      std::sqrt(std::numeric_limits<double>::epsilon()) * scale;
  return std::max(error, poleSeparation);
}

double strictThreshold(double lam2, double error) {
  return std::nextafter(lam2 + numericalResolution(lam2, error),
                        std::numeric_limits<double>::infinity());
}

double aboveSpectrumPole(const std::vector<double>& vals, double threshold) {
  while (std::binary_search(vals.begin(), vals.end(), threshold))
    threshold = std::nextafter(threshold,
                               std::numeric_limits<double>::infinity());
  return threshold;
}

bool firstCoupledImprovement(const Graph& anchor,
                             const std::vector<double>& vals,
                             const std::vector<double>& vecs, double error,
                             const std::vector<uint64_t>* forbiddenState,
                             int proposalWidth, bool includeCoupling,
                             Graph& improved) {
  const detail::GroundedInverse inverse(
      anchor.pix(), anchor.edges(), 0, 0, 0.0);
  const CoupledDomain domain =
      buildCoupledDomain(anchor, inverse, nullptr, nullptr, proposalWidth);
  const double threshold =
      aboveSpectrumPole(vals, strictThreshold(vals[1], error));
  const eig::ShiftedSwapInertia decision(vals, vecs, threshold);
  const PairIndex& pix = anchor.pix();
  const SwapConnectivity connectivity(anchor);
  std::vector<eig::ShiftedSwapInertia::PairProjection> projections(
      static_cast<size_t>(pix.numPairs()));
  for (int pair : domain.additions)
    projections[pair] = decision.projectPair(pix.i(pair), pix.j(pair));
  for (int pair : domain.removals)
    projections[pair] = decision.projectPair(pix.i(pair), pix.j(pair));

  // If the forbidden graph is one swap from this anchor, identify that swap
  // once. Comparing a full O(n^2)-bit state for every passing candidate would
  // turn the streaming scan into a hidden O(n^5) path.
  Swap forbiddenSwap{-1, -1};
  bool forbiddenIsOneSwap = false;
  if (forbiddenState) {
    int differences = 0;
    for (int pair = 0; pair < anchor.numPairs(); ++pair) {
      const bool other =
          ((*forbiddenState)[static_cast<size_t>(pair) >> 6] >> (pair & 63)) & 1ULL;
      if (anchor.has(pair) == other) continue;
      ++differences;
      if (other) forbiddenSwap.add = pair;
      else forbiddenSwap.remove = pair;
    }
    forbiddenIsOneSwap = differences == 2 && forbiddenSwap.add >= 0 &&
                         forbiddenSwap.remove >= 0;
  }

  bool found = false;
  double bestScore = -std::numeric_limits<double>::infinity();
  Swap bestSwap{-1, -1};
  auto precedes = [](double score, int add, int remove,
                     bool selectionExists, double selectedScore,
                     Swap selectedSwap) {
    return !selectionExists || score > selectedScore ||
           (score == selectedScore &&
            (add < selectedSwap.add ||
             (add == selectedSwap.add && remove < selectedSwap.remove)));
  };
  auto consider = [&](int add, int remove, bool& localFound,
                      double& localScore, Swap& localSwap) {
    if (forbiddenIsOneSwap && add == forbiddenSwap.add &&
        remove == forbiddenSwap.remove)
      return;
    const Swap candidate{add, remove};
    if (!connectivity.keepsConnected(candidate)) return;
    const double score = proposalScore(
        anchor, inverse, domain, add, remove, includeCoupling);
    // Once a passing candidate exists, an inferior score cannot change the
    // exact sorted-order winner. Check the cheap total order first so the
    // inertia determinant is evaluated only for record candidates.
    if (!precedes(score, add, remove,
                  localFound, localScore, localSwap))
      return;
    if (!decision.lam2AtLeast(projections[add], projections[remove])) return;
    localFound = true;
    localScore = score;
    localSwap = candidate;
  };

#ifdef _OPENMP
  // Completed-swap decisions are independent. Static OpenMP fibers use idle
  // cores without changing the candidate set, score order, or eigensolve
  // budget. Thread-local exact winners merge under the same total order, so
  // the result is bit-for-bit deterministic for every team size.
#pragma omp parallel
  {
    bool localFound = false;
    double localScore = -std::numeric_limits<double>::infinity();
    Swap localSwap{-1, -1};
    const int addRows = static_cast<int>(
        std::min(domain.rank, domain.additions.size()));
    const int removalColumns = static_cast<int>(
        std::min(domain.rank, domain.removals.size()));
#pragma omp for schedule(static) nowait
    for (int ai = 0; ai < addRows; ++ai)
      for (size_t ri = 0; ri < domain.removals.size(); ++ri)
        consider(domain.additions[ai], domain.removals[ri],
                 localFound, localScore, localSwap);
#pragma omp for schedule(static) nowait
    for (int ai = addRows;
         ai < static_cast<int>(domain.additions.size()); ++ai)
      for (int ri = 0; ri < removalColumns; ++ri)
        consider(domain.additions[ai], domain.removals[ri],
                 localFound, localScore, localSwap);
#pragma omp critical(ours_coupled_winner)
    {
      if (localFound &&
          precedes(localScore, localSwap.add, localSwap.remove,
                   found, bestScore, bestSwap)) {
        found = true;
        bestScore = localScore;
        bestSwap = localSwap;
      }
    }
  }
#else
  visitRankMatchedUnion(
      anchor, inverse, domain, includeCoupling,
      [&](size_t, size_t, int add, int remove, double) {
        consider(add, remove, found, bestScore, bestSwap);
      });
#endif
  if (!found) return false;
  improved = anchor;
  improved.applySwap(bestSwap);
  return true;
}

bool leastDipProposal(
    const Graph& anchor, const std::vector<double>& vals,
    const std::vector<double>& vecs, double error,
    const std::vector<Swap>& proposals, const std::vector<char>* tried,
    Swap& selected, size_t* selectedIndex = nullptr) {
  const double resolution = numericalResolution(vals[1], error);
  std::vector<size_t> candidates;
  candidates.reserve(proposals.size());
  const SwapConnectivity connectivity(anchor);
  for (size_t index = 0; index < proposals.size(); ++index) {
    if (tried && (*tried)[index]) continue;
    const Swap swap = proposals[index];
    if (!connectivity.keepsConnected(swap)) continue;
    candidates.push_back(index);
  }
  if (candidates.empty()) return false;

  // All roots share the same interlacing bracket. Parallel bisection builds
  // one shifted resolvent per level and scans every root with O(1) rank-two
  // inertia tests. It therefore finds the least dip in O(n^3) work instead of
  // performing a length-n secular sum independently for O(n^3) candidates.
  double lower = 0.0;
  double upper = std::nextafter(
      vals[std::min<size_t>(2, vals.size() - 1)],
      std::numeric_limits<double>::infinity());
  const PairIndex& pix = anchor.pix();
  // lambda_max(L)<=n and n is represented by int, while resolution is at
  // least sqrt(machine epsilon). This machine-derived count is therefore a
  // strict bound on every binary refinement; there is no data-dependent or
  // asymptotically growing iteration cap hidden in the cubic scan.
  constexpr int maxBisections =
      std::numeric_limits<int>::digits +
      (std::numeric_limits<double>::digits + 1) / 2 + 1;
  for (int iteration = 0;
       iteration < maxBisections && upper - lower > resolution;
       ++iteration) {
    double midpoint = lower + (upper - lower) / 2.0;
    if (midpoint == lower || midpoint == upper) break;
    midpoint = aboveSpectrumPole(vals, midpoint);
    if (midpoint == upper) break;
    const eig::ShiftedSwapInertia decision(vals, vecs, midpoint);
    bool anyAtLeast = false;
    for (size_t index : candidates) {
      const Swap swap = proposals[index];
      if (decision.lam2AtLeast(
              pix.i(swap.add), pix.j(swap.add),
              pix.i(swap.remove), pix.j(swap.remove))) {
        anyAtLeast = true;
        break;
      }
    }
    if (anyAtLeast)
      lower = midpoint;
    else
      upper = midpoint;
  }

  bool found = false;
  size_t foundIndex = 0;
  if (lower == 0.0) {
    // Every connected root lies in the first numerical bin.
    for (size_t index : candidates) {
      const Swap swap = proposals[index];
      if (!found || swap.add < selected.add ||
          (swap.add == selected.add && swap.remove < selected.remove)) {
        selected = swap;
        foundIndex = index;
        found = true;
      }
    }
  } else {
    const eig::ShiftedSwapInertia decision(vals, vecs, lower);
    for (size_t index : candidates) {
      const Swap swap = proposals[index];
      if (!decision.lam2AtLeast(
              pix.i(swap.add), pix.j(swap.add),
              pix.i(swap.remove), pix.j(swap.remove)))
        continue;
      if (!found || swap.add < selected.add ||
          (swap.add == selected.add && swap.remove < selected.remove)) {
        selected = swap;
        foundIndex = index;
        found = true;
      }
    }
  }
  if (found && selectedIndex) *selectedIndex = foundIndex;
  return found;
}

// Two computed eigenvalues belong to the unresolved lambda_2 cluster exactly
// when their residual intervals overlap. The factor two is not a parameter:
// each endpoint carries one copy of the same backward-error radius.
int resolvedMultiplicity(const std::vector<double>& vals, double error) {
  if (vals.size() < 2) return 0;
  int multiplicity = 1;
  for (size_t i = 2; i < vals.size(); ++i) {
    if (vals[i] - error <= vals[1] + error)
      ++multiplicity;
    else
      break;
  }
  return multiplicity;
}

double globalFiedlerBound(const Graph& g) {
  if (g.m() == g.numPairs()) return static_cast<double>(g.n());
  return static_cast<double>((2 * g.m()) / g.n());
}

std::vector<Swap> coupledProposalsImpl(
    const Graph& g, const detail::GroundedInverse& inverse,
    const std::vector<char>* allowedAdds,
    const std::vector<char>* allowedRemovals, bool responsesOnly,
    int proposalWidth, bool includeCoupling,
    size_t* distinctPairsExamined) {
  if (distinctPairsExamined) *distinctPairsExamined = 0;
  const CoupledDomain domain =
      buildCoupledDomain(
          g, inverse, allowedAdds, allowedRemovals, proposalWidth);
  const std::vector<int>& additions = domain.additions;
  const std::vector<int>& removals = domain.removals;
  if (additions.empty() || removals.empty()) return {};

  std::vector<ScoredSwap> scored;
  if (!responsesOnly) {
    // The resistance Gram acts on 1-perp, whose intrinsic dimension is n-1.
    // Use that dimension as the non-fitted row/column budget: each retained
    // marginal extreme is coupled with every possible partner. The union
    // contains O(n^3) completed swaps and is formed in eigensolve-order work.
    const size_t addRows = std::min(domain.rank, additions.size());
    const size_t removalColumns = std::min(domain.rank, removals.size());
    scored.reserve(addRows * removals.size() +
                   removalColumns * additions.size());
    visitRankMatchedUnion(
        g, inverse, domain, includeCoupling,
        [&](size_t, size_t, int add, int remove, double score) {
          scored.push_back({score, Swap{add, remove}});
        },
        distinctPairsExamined);
  } else {
    std::set<std::pair<int, int>> closure;
    // Depth continuations retain the exact bidirectional best-response closure
    // of the same rank-matched row/column union. Every coordinate has a
    // retained partner, the union maximizer is present, and at most A+R roots
    // survive. No extra baseline pair or fallback is injected.
    std::vector<int> bestRemovalForAdd(additions.size(), -1);
    std::vector<double> bestRemovalScore(
        additions.size(), -std::numeric_limits<double>::infinity());
    std::vector<int> bestAddForRemoval(removals.size(), -1);
    std::vector<double> bestAddScore(
        removals.size(), -std::numeric_limits<double>::infinity());
    visitRankMatchedUnion(
        g, inverse, domain, includeCoupling,
        [&](size_t ai, size_t ri, int, int, double score) {
          if (score > bestRemovalScore[ai] ||
              (score == bestRemovalScore[ai] &&
               removals[ri] < bestRemovalForAdd[ai])) {
            bestRemovalScore[ai] = score;
            bestRemovalForAdd[ai] = removals[ri];
          }
          if (score > bestAddScore[ri] ||
              (score == bestAddScore[ri] &&
               additions[ai] < bestAddForRemoval[ri])) {
            bestAddScore[ri] = score;
            bestAddForRemoval[ri] = additions[ai];
          }
        },
        distinctPairsExamined);
    for (size_t ai = 0; ai < additions.size(); ++ai)
      closure.insert({additions[ai], bestRemovalForAdd[ai]});
    for (size_t ri = 0; ri < removals.size(); ++ri)
      closure.insert({bestAddForRemoval[ri], removals[ri]});
    scored.reserve(closure.size());
    for (const auto& [add, remove] : closure)
      scored.push_back(
          {proposalScore(g, inverse, domain, add, remove, includeCoupling),
           Swap{add, remove}});
  }
  radixSortScored(scored);

  std::vector<Swap> result;
  result.reserve(scored.size());
  for (const ScoredSwap& candidate : scored) result.push_back(candidate.swap);
  return result;
}

bool applyWithInverse(Graph& graph, detail::GroundedInverse& inverse, Swap swap) {
  const PairIndex& pix = graph.pix();
  Graph trial = graph;
  trial.applySwap(swap);
  if (!inverse.tryAddEdge(pix.i(swap.add), pix.j(swap.add))) return false;
  if (!inverse.tryRemoveEdge(pix.i(swap.remove), pix.j(swap.remove)))
    return false;
  graph = std::move(trial);
  return true;
}

void editCoordinates(const PairIndex& pix, const std::vector<double>& vecs,
                     int n, const std::vector<Swap>& edits,
                     std::vector<double>& coordinates,
                     std::vector<int>& signs) {
  coordinates.resize(static_cast<size_t>(2 * edits.size()) * n);
  signs.resize(2 * edits.size());
  for (size_t i = 0; i < edits.size(); ++i) {
    const int pairs[2] = {edits[i].add, edits[i].remove};
    for (int side = 0; side < 2; ++side) {
      const int pair = pairs[side];
      const int u = pix.i(pair), v = pix.j(pair);
      const size_t row = 2 * i + static_cast<size_t>(side);
      for (int k = 0; k < n; ++k)
        coordinates[row * n + k] =
            vecs[static_cast<size_t>(u) * n + k] -
            vecs[static_cast<size_t>(v) * n + k];
      signs[row] = side == 0 ? +1 : -1;
    }
  }
}

bool clearsAnchor(const Graph& anchor, const std::vector<double>& vals,
                  const std::vector<double>& vecs, double error,
                  const std::vector<Swap>& edits) {
  std::vector<double> coordinates;
  std::vector<int> signs;
  editCoordinates(anchor.pix(), vecs, anchor.n(), edits, coordinates, signs);
  const double threshold =
      aboveSpectrumPole(vals, strictThreshold(vals[1], error));
  return eig::lam2AfterEditAtLeast(vals, coordinates, signs, threshold);
}

// Build one complete multiplicity-radius path from the paid anchor. The root
// comes from the coupled oracle's bidirectional response closure. All later
// coordinates are consumed in a parameter-free coupled coordinate step on the
// current prefix. Each step
// examines the complete row of the strongest remaining resistance addition
// and the complete column of the weakest remaining resistance removal, then
// takes the exact coupled-volume maximizer of their union. Those are the two
// canonical marginal coordinates of the add/remove product space, not a
// fitted shortlist. Effective resistances and connectivity are O(n^2) per
// level and inverse updates are O(n^2); r<=n-1 therefore makes the whole path
// O(n^3). The final rank-2r inertia decision is also O(n^3).
bool findCubicDepthEscape(
    const Graph& anchor, const std::vector<double>& vals,
    const std::vector<double>& vecs, double error, int radius, Swap root,
    bool includeCoupling, Graph& escaped) {
  const int pairCount = anchor.numPairs();
  if (radius > std::min(anchor.m(), pairCount - anchor.m())) return false;

  Graph path = anchor;
  detail::GroundedInverse inverse(
      anchor.pix(), anchor.edges(), 0, 0, 0.0);
  if (!applyWithInverse(path, inverse, root)) return false;
  std::vector<Swap> edits{root};
  std::vector<char> usedAdds(pairCount, 0), usedRemovals(pairCount, 0);
  usedAdds[root.add] = 1;
  usedRemovals[root.remove] = 1;
  size_t coordinatePairsExamined = 0;

  while (static_cast<int>(edits.size()) < radius) {
    std::vector<double> resistance(pairCount, 0.0);
    int pivotAdd = -1, pivotRemoval = -1;
    for (int pair = 0; pair < pairCount; ++pair) {
      const bool eligibleAdd =
          !anchor.has(pair) && !usedAdds[pair] && !path.has(pair);
      const bool eligibleRemoval =
          anchor.has(pair) && !usedRemovals[pair] && path.has(pair);
      if (!eligibleAdd && !eligibleRemoval) continue;
      resistance[pair] = inverse.effectiveResistance(
          anchor.pix().i(pair), anchor.pix().j(pair));
      if (eligibleAdd &&
          (pivotAdd < 0 || resistance[pair] > resistance[pivotAdd] ||
           (resistance[pair] == resistance[pivotAdd] && pair < pivotAdd)))
        pivotAdd = pair;
      if (eligibleRemoval &&
          (pivotRemoval < 0 || resistance[pair] < resistance[pivotRemoval] ||
           (resistance[pair] == resistance[pivotRemoval] &&
            pair < pivotRemoval)))
        pivotRemoval = pair;
    }
    if (pivotAdd < 0 || pivotRemoval < 0) return false;

    const SwapConnectivity connectivity(path);
    bool found = false;
    double bestScore = -std::numeric_limits<double>::infinity();
    Swap next{-1, -1};
    auto consider = [&](int add, int remove) {
      ++coordinatePairsExamined;
      const Swap candidate{add, remove};
      if (!connectivity.keepsConnected(candidate)) return;
      double score = resistance[add] - resistance[remove];
      if (includeCoupling) {
        const double transfer = inverse.transfer(
            anchor.pix().i(add), anchor.pix().j(add),
            anchor.pix().i(remove), anchor.pix().j(remove));
        score += -resistance[add] * resistance[remove] + transfer * transfer;
      }
      if (!found || score > bestScore ||
          (score == bestScore &&
           (add < next.add || (add == next.add && remove < next.remove)))) {
        found = true;
        bestScore = score;
        next = candidate;
      }
    };
    for (int pair = 0; pair < pairCount; ++pair) {
      if (anchor.has(pair) && !usedRemovals[pair] && path.has(pair))
        consider(pivotAdd, pair);
      if (!anchor.has(pair) && !usedAdds[pair] && !path.has(pair))
        consider(pair, pivotRemoval);
    }
    if (!found) return false;
    if (!applyWithInverse(path, inverse, next)) return false;
    edits.push_back(next);
    usedAdds[next.add] = 1;
    usedRemovals[next.remove] = 1;
  }
  if (coordinatePairsExamined >
      OURS::cubicContinuationPairBound(anchor, radius))
    throw std::logic_error("cubic continuation exceeded structural pair bound");
  if (!clearsAnchor(anchor, vals, vecs, error, edits)) return false;
  path.checkFeasible();
  escaped = std::move(path);
  return true;
}

// Pay for one deliberately non-improving branch, then use that branch's full
// eigensystem to decide the completed two-swap recovery. Branches are finite,
// ordered by their exact secular lambda_2 lower bound, and persistent across
// visits. One visit expands exactly one branch -- the indivisible DFS action.
bool findTwoStepEscape(
    const Graph& anchor, const std::vector<double>& anchorVals,
    double anchorError, Swap branch,
    Oracle& oracle, int proposalWidth, bool includeCoupling,
    Graph& escaped, bool& escapedIsPaid, std::vector<double>& escapedVals,
    std::vector<double>& escapedVecs, double& escapedError,
    int& escapedMoves) {
  const double anchorLower = anchorVals[1] - anchorError;
  const double anchorUpper = anchorVals[1] + anchorError;

  if (oracle.exhausted()) return false;
  Graph intermediate = anchor;
  intermediate.applySwap(branch);
  std::vector<double> vals, vecs;
  oracle.eigensystem(intermediate, vals, vecs);
  const double error = eigensystemError(intermediate, vals, vecs);

  auto continueFromIntermediate = [&]() {
    escaped = std::move(intermediate);
    escapedIsPaid = true;
    escapedVals = std::move(vals);
    escapedVecs = std::move(vecs);
    escapedError = error;
    escapedMoves = 1;
    return true;
  };

  if (vals[1] > anchorVals[1]) {
    return continueFromIntermediate();
  }
  if (oracle.exhausted()) return continueFromIntermediate();

  // Rank-two interlacing: a further one-swap recovery cannot lift lambda_2
  // above this intermediate's lambda_3. Reject only when the residual
  // intervals prove the branch dead.
  if (vals.size() > 2 && vals[2] + error <= anchorLower)
    return continueFromIntermediate();

  const detail::GroundedInverse recoveryInverse(
      intermediate.pix(), intermediate.edges(), 0, 0, 0.0);
  const std::vector<Swap> recoveries = coupledProposalsImpl(
      intermediate, recoveryInverse, nullptr, nullptr, false,
      proposalWidth, includeCoupling);
  const double recoveryThreshold =
      aboveSpectrumPole(vals, strictThreshold(anchorUpper, error));
  const eig::ShiftedSwapInertia decision(vals, vecs, recoveryThreshold);
  const PairIndex& pix = intermediate.pix();
  const SwapConnectivity recoveryConnectivity(intermediate);
  for (Swap recovery : recoveries) {
    if (!recoveryConnectivity.keepsConnected(recovery)) continue;
    if (!decision.lam2AtLeast(
            pix.i(recovery.add), pix.j(recovery.add),
            pix.i(recovery.remove), pix.j(recovery.remove)))
      continue;
    if (oracle.exhausted()) return false;
    escaped = intermediate;
    escaped.applySwap(recovery);
    escapedIsPaid = false;
    escapedVals.clear();
    escapedVecs.clear();
    escapedError = std::numeric_limits<double>::infinity();
    escapedMoves = 2;
    return true;
  }
  // No one-step recovery clears the anchor. Preserve the paid branch and let
  // the normal climb continue deeper instead of discarding the DFS prefix.
  return continueFromIntermediate();
}

}  // namespace

std::vector<Swap> OURS::coupledProposals(const Graph& graph) {
  // No ridge and no periodic-update policy: this inverse is factored once and
  // used read-only to define the proposal order.
  const detail::GroundedInverse inverse(graph.pix(), graph.edges(), 0, 0, 0.0);
  return coupledProposalsImpl(
      graph, inverse, nullptr, nullptr, true, 0, true);
}

size_t OURS::coupledProposalPairsExamined(const Graph& graph) {
  const detail::GroundedInverse inverse(graph.pix(), graph.edges(), 0, 0, 0.0);
  size_t examined = 0;
  static_cast<void>(coupledProposalsImpl(
      graph, inverse, nullptr, nullptr, true, 0, true, &examined));
  return examined;
}

size_t OURS::cubicContinuationPairBound(
    const Graph& graph, int radius) {
  if (radius <= 1) return 0;
  if (radius > std::min(
          graph.m(), graph.numPairs() - graph.m()))
    return 0;
  const size_t r = static_cast<size_t>(radius);
  return (r - 1) * (static_cast<size_t>(graph.numPairs()) - r);
}

Result OURS::run(Graph start, Oracle& oracle, Rng& rng) const {
  if (cfg_.proposalWidth < 0)
    throw std::invalid_argument("proposal width must be nonnegative");
  if (!cfg_.inertiaScreen && cfg_.attack)
    throw std::invalid_argument(
        "the no-inertia ablation is defined only with depth disabled");
  const auto begin = std::chrono::steady_clock::now();
  const PairIndex& pix = start.pix();
  const int m = start.m();

  Graph current = start;
  Graph best = start;
  double bestLam2 = -std::numeric_limits<double>::infinity();
  double bestError = std::numeric_limits<double>::infinity();
  std::vector<double> bestVals, bestVecs;
  std::vector<double> vals, vecs;
  double error = std::numeric_limits<double>::infinity();
  bool currentIsPaid = false;
  bool depthExpandedThisBasin = false;
  std::vector<uint64_t> forbiddenReturn;
  bool depthRootsBuilt = false;
  std::vector<Swap> depthRoots;
  std::vector<char> triedDepthRoots;
  bool compositeRootsBuilt = false;
  std::vector<Swap> compositeRoots;
  size_t compositeRootCursor = 0;
  int steps = 0;

  auto recordBest = [&](const Graph& graph,
                        const std::vector<double>& graphVals,
                        const std::vector<double>& graphVecs,
                        double graphError) {
    if (!bestVals.empty() && graphVals[1] <= bestLam2) return;
    best = graph;
    bestLam2 = graphVals[1];
    bestError = graphError;
    bestVals = graphVals;
    bestVecs = graphVecs;
    depthRootsBuilt = false;
    depthRoots.clear();
    triedDepthRoots.clear();
    compositeRootsBuilt = false;
    compositeRoots.clear();
    compositeRootCursor = 0;
    depthExpandedThisBasin = false;
    forbiddenReturn.clear();
  };

  // A depth-root evaluation can consume the final budget unit inside
  // findTwoStepEscape().  Process that already-paid state once more so it can
  // update the best queried graph before terminating; never issue a new solve after
  // exhaustion.
  while (currentIsPaid || !oracle.exhausted()) {
    if (!currentIsPaid) {
      oracle.eigensystem(current, vals, vecs);
      error = eigensystemError(current, vals, vecs);
      currentIsPaid = true;
    }

    recordBest(current, vals, vecs, error);

    // The last paid solve may improve the best queried graph, but no unevaluated graph
    // is reported after B is exhausted.
    if (oracle.exhausted()) break;
    // A graph-theoretic upper bound certifies global optimality only when the
    // LOWER endpoint of the computed lambda_2 interval reaches it. Using the
    // upper endpoint could stop on a graph whose true lambda_2 is still below
    // the bound. This is a certificate, not a numerical closeness test.
    if (bestLam2 - bestError >= globalFiedlerBound(best)) break;

    // Coupled tree-volume supplies only the proposal order. The complete
    // add/remove decision is the rank-two inertia test from this paid
    // eigensystem. Exact connectivity is checked before the numerical inertia
    // screen; exhausting the finite stream defines a one-swap stall in the
    // rank-matched proposal neighborhood.
    if (cfg_.inertiaScreen) {
      Graph climbed = current;
      const std::vector<uint64_t>* forbidden =
          forbiddenReturn.empty() ? nullptr : &forbiddenReturn;
      if (firstCoupledImprovement(
              current, vals, vecs, error, forbidden, cfg_.proposalWidth,
              cfg_.includeCoupling, climbed)) {
        current = std::move(climbed);
        currentIsPaid = false;
        ++steps;
        continue;
      }
    } else {
      // Clean screening ablation. Preserve the same connected proposal set
      // and score order, but pay for candidates one by one until a full solve
      // verifies an improvement. Rejected solves still count toward B and
      // toward the best queried graph.
      const detail::GroundedInverse inverse(
          pix, current.edges(), 0, 0, 0.0);
      const std::vector<Swap> proposals = coupledProposalsImpl(
          current, inverse, nullptr, nullptr, false, cfg_.proposalWidth,
          cfg_.includeCoupling);
      const SwapConnectivity connectivity(current);
      const double threshold = strictThreshold(vals[1], error);
      bool climbed = false;
      for (Swap swap : proposals) {
        if (!connectivity.keepsConnected(swap)) continue;
        if (oracle.exhausted()) break;
        Graph trial = current;
        trial.applySwap(swap);
        std::vector<double> trialVals, trialVecs;
        oracle.eigensystem(trial, trialVals, trialVecs);
        const double trialError =
            eigensystemError(trial, trialVals, trialVecs);
        recordBest(trial, trialVals, trialVecs, trialError);
        if (trialVals[1] < threshold) continue;
        current = std::move(trial);
        vals = std::move(trialVals);
        vecs = std::move(trialVecs);
        error = trialError;
        currentIsPaid = true;
        climbed = true;
        ++steps;
        break;
      }
      if (climbed) continue;
    }

    if (cfg_.attack && !depthExpandedThisBasin) {
      const int multiplicity = resolvedMultiplicity(bestVals, bestError);
      // An unresolved numerical cluster supplies a conservative depth radius,
      // not an exact multiplicity certificate. If that radius is infeasible,
      // skip this depth visit and keep the budgeted breadth search alive; only
      // exact arithmetic permits the corresponding global-optimality theorem.
      const bool radiusFeasible =
          multiplicity <= std::min(best.m(), best.numPairs() - best.m());

      Swap branch{-1, -1};
      bool hasBranch = false;

      if (radiusFeasible && multiplicity <= 2) {
        if (!depthRootsBuilt) {
          const detail::GroundedInverse inverse(
              pix, best.edges(), 0, 0, 0.0);
          depthRoots = coupledProposalsImpl(
              best, inverse, nullptr, nullptr, false, cfg_.proposalWidth,
              cfg_.includeCoupling);
          triedDepthRoots.assign(depthRoots.size(), 0);
          depthRootsBuilt = true;
        }
        size_t branchIndex = 0;
        hasBranch = leastDipProposal(
            best, bestVals, bestVecs, bestError, depthRoots,
            &triedDepthRoots, branch, &branchIndex);
        if (hasBranch) triedDepthRoots[branchIndex] = 1;
        if (hasBranch) {
          Graph escaped = current;
          bool escapedIsPaid = false;
          std::vector<double> escapedVals, escapedVecs;
          double escapedError = std::numeric_limits<double>::infinity();
          int escapedMoves = 0;
          if (findTwoStepEscape(
                  best, bestVals, bestError, branch, oracle,
                  cfg_.proposalWidth, cfg_.includeCoupling,
                  escaped, escapedIsPaid, escapedVals,
                  escapedVecs, escapedError, escapedMoves)) {
            const bool continuesBelowIncumbent =
                escapedIsPaid && escapedVals[1] <= bestLam2;
            current = std::move(escaped);
            currentIsPaid = escapedIsPaid;
            depthExpandedThisBasin = true;
            if (continuesBelowIncumbent)
              forbiddenReturn = best.edges().words();
            else
              forbiddenReturn.clear();
            if (currentIsPaid) {
              vals = std::move(escapedVals);
              vecs = std::move(escapedVecs);
              error = escapedError;
            }
            steps += escapedMoves;
            continue;
          }
        }
      }

      if (radiusFeasible && multiplicity > 2) {
        if (!compositeRootsBuilt) {
          const detail::GroundedInverse inverse(
              pix, best.edges(), 0, 0, 0.0);
          compositeRoots = coupledProposalsImpl(
              best, inverse, nullptr, nullptr, true, cfg_.proposalWidth,
              cfg_.includeCoupling);
          compositeRootsBuilt = true;
        }
        const SwapConnectivity connectivity(best);
        while (compositeRootCursor < compositeRoots.size()) {
          branch = compositeRoots[compositeRootCursor++];
          if (connectivity.keepsConnected(branch)) {
            hasBranch = true;
            break;
          }
        }
      }

      if (radiusFeasible && multiplicity > 2 && hasBranch) {
        // In exact arithmetic multiplicity is the minimum useful complete
        // radius by interlacing.  Here the residual-overlap cluster is the
        // conservative, resolution-aware replacement.  One monotone proposal
        // scan plus one full-edit inertia decision keeps the attempt cubic.
        const int radius = multiplicity;
        Graph escaped = best;
        if (findCubicDepthEscape(
                best, bestVals, bestVecs, bestError, radius, branch,
                cfg_.includeCoupling, escaped)) {
          current = std::move(escaped);
          currentIsPaid = false;
          steps += radius;
          continue;
        }
      }
    }

    if (!cfg_.restart) break;
    // Use the same random-spanning-tree restart distribution as the repository
    // ER/FV search baselines. The path-seeded graph is the one shared initial
    // condition, not their restart law.
    current = Graph::randomConnected(pix, m, rng);
    currentIsPaid = false;
    depthExpandedThisBasin = false;
    forbiddenReturn.clear();
  }

  best.checkFeasible();
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  const int multiplicity =
      bestVals.empty() ? 0 : resolvedMultiplicity(bestVals, bestError);
  return Result{name(),           best.n(),       best.m(), bestLam2,
                best.edgeList(), oracle.calls(), oracle.queries(),
                wall,             0,              steps,    best.momentUpperBound(),
                multiplicity};
}

}  // namespace algconn

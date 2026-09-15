// Tape comparison: win/tie/loss between two result tapes, and hit rate against
// an enumerated optimum. This is the analysis path for the paper's claims --
// no Python is used anywhere outside figure generation.
//
//   ./compare --a ours.csv --b results/exp08_sweeps/er_m_n32.csv
//   ./compare --a ours.csv --b er.csv --opt results/exp11_enum_n12/optimal_n12.csv
//   ./compare --a ours_b_n64.csv --b er_b_n64.csv --curve
//
// Two tape shapes are recognized automatically:
//   m-sweep      keyed by (m, seed), value column lam2
//   budget sweep keyed by (seed, calls), value column best_lam2; --curve reports
//                the running best at each decade and the per-seed comparison.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Table {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
  int col(const std::string& name) const {
    for (size_t i = 0; i < header.size(); ++i)
      if (header[i] == name) return (int)i;
    return -1;
  }
};

Table readCsv(const std::string& path) {
  Table t;
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::string line;
  bool first = true;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> f;
    std::string cell;
    std::istringstream ss(line);
    while (std::getline(ss, cell, ',')) f.push_back(cell);
    if (first) { t.header = f; first = false; }
    else t.rows.push_back(std::move(f));
  }
  return t;
}

double med(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

using Key = std::pair<long, long>;

// Best value per key. For m-sweeps the key is (m, seed) and one row is one run.
// For budget sweeps the key is (seed, 0) and the value is the running best at
// or below `cap` evaluations.
std::map<Key, double> collapse(const Table& t, bool curve, long cap,
                               const std::string& method) {
  std::map<Key, double> out;
  const int cm = t.col("m"), cs = t.col("seed"), cmeth = t.col("method");
  const int cl = curve ? t.col("best_lam2") : t.col("lam2");
  const int cc = t.col("calls");
  if (cl < 0 || cs < 0) {
    std::fprintf(stderr, "tape lacks the expected columns\n");
    std::exit(1);
  }
  for (const auto& r : t.rows) {
    if (!method.empty() && cmeth >= 0 && r[cmeth] != method) continue;
    const long seed = std::stol(r[cs]);
    double v = std::stod(r[cl]);
    Key k;
    if (curve) {
      if (cc >= 0 && cap > 0 && std::stol(r[cc]) > cap) continue;
      k = {seed, 0};
      auto it = out.find(k);
      if (it != out.end()) v = std::max(it->second, v);
    } else {
      k = {cm >= 0 ? std::stol(r[cm]) : 0, seed};
    }
    out[k] = v;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string aPath, bPath, optPath, aMeth, bMeth;
  bool curve = false, perM = false;
  for (int i = 1; i < argc; ++i) {
    auto nxt = [&]() { return std::string(argv[++i]); };
    const std::string a = argv[i];
    if (a == "--a") aPath = nxt();
    else if (a == "--b") bPath = nxt();
    else if (a == "--opt") optPath = nxt();
    else if (a == "--curve") curve = true;
    else if (a == "--per-m") perM = true;
    else if (a == "--a-method") aMeth = nxt();
    else if (a == "--b-method") bMeth = nxt();
  }
  if (aPath.empty()) { std::fprintf(stderr, "need --a\n"); return 1; }

  const Table A = readCsv(aPath);
  const std::vector<long> caps =
      curve ? std::vector<long>{10, 100, 1000, 10000, 100000, 1000000}
            : std::vector<long>{0};

  // Optional: hit rate against the enumerated optimum, keyed by m.
  std::map<long, double> opt;
  if (!optPath.empty()) {
    const Table O = readCsv(optPath);
    int cm = O.col("m"), cv = O.col("lam2_max");
    if (cv < 0) cv = O.col("lam2");
    for (const auto& r : O.rows) opt[std::stol(r[cm])] = std::stod(r[cv]);
  }

  for (long cap : caps) {
    const auto av = collapse(A, curve, cap, aMeth);
    if (av.empty()) continue;
    std::string tag = curve ? ("B<=" + std::to_string(cap)) : std::string("all");
    std::printf("== %s ==  A=%s (%zu runs)\n", tag.c_str(), aPath.c_str(), av.size());

    if (!opt.empty()) {
      long hit = 0, tot = 0;
      for (const auto& [k, v] : av) {
        auto it = opt.find(k.first);
        if (it == opt.end()) continue;
        ++tot;
        if (v >= it->second - 1e-9) ++hit;
      }
      if (tot)
        std::printf("   hit rate vs enumeration: %ld/%ld = %.1f%%\n", hit, tot,
                    100.0 * (double)hit / (double)tot);
    }

    if (bPath.empty()) continue;
    const Table B = readCsv(bPath);
    const auto bv = collapse(B, curve, cap, bMeth);
    long w = 0, t = 0, l = 0;
    double relSum = 0.0;
    long relN = 0;
    std::vector<std::pair<double, Key>> losses;
    std::vector<double> avm, bvm;
    for (const auto& [k, v] : av) {
      auto it = bv.find(k);
      if (it == bv.end()) continue;
      const double d = v - it->second;
      avm.push_back(v);
      bvm.push_back(it->second);
      if (d > 1e-9) ++w;
      else if (d < -1e-9) { ++l; losses.push_back({d, k}); }
      else ++t;
      if (it->second > 1e-12) { relSum += d / it->second; ++relN; }
    }
    std::printf("   vs %s : %ldW/%ldT/%ldL   median A %.4f vs B %.4f   mean margin %+.1f%%\n",
                bPath.c_str(), w, t, l, med(avm), med(bvm),
                relN ? 100.0 * relSum / (double)relN : 0.0);
    std::sort(losses.begin(), losses.end());
    for (size_t i = 0; i < losses.size() && i < 6; ++i)
      std::printf("      loss: m=%ld seed=%ld  %+.4f\n", losses[i].second.first,
                  losses[i].second.second, losses[i].first);

    if (perM) {
      // One row per m: medians, per-seed W/T/L, hit counts against the
      // enumerated optimum, and the multiplicity of that optimum -- the
      // quantity Theorem 1 prices the escape by.
      std::map<long, std::vector<double>> A_m, B_m;
      for (const auto& [k, v] : av) A_m[k.first].push_back(v);
      for (const auto& [k, v] : bv) B_m[k.first].push_back(v);
      std::printf("\n  m   opt      ours_med  er_med   W/T/L      ours_hit er_hit  gap%%\n");
      for (const auto& [m, va] : A_m) {
        std::vector<double> vb = B_m.count(m) ? B_m[m] : std::vector<double>{};
        long mw = 0, mt = 0, ml = 0, ah = 0, bh = 0, nn = 0;
        for (const auto& [k, v] : av) {
          if (k.first != m) continue;
          auto it = bv.find(k);
          if (it == bv.end()) continue;
          ++nn;
          const double d = v - it->second;
          if (d > 1e-9) ++mw; else if (d < -1e-9) ++ml; else ++mt;
          auto o = opt.find(m);
          if (o != opt.end()) {
            if (v >= o->second - 1e-9) ++ah;
            if (it->second >= o->second - 1e-9) ++bh;
          }
        }
        const double om = med(va), bm2 = med(vb);
        auto o = opt.find(m);
        const double ov = o != opt.end() ? o->second : 0.0;
        const double gap = ov > 1e-12 ? 100.0 * (ov - om) / ov : 0.0;
        std::printf("  %-3ld %-8.4f %-9.4f %-8.4f %3ld/%3ld/%-3ld  %3ld/%-3ld  %3ld/%-3ld %6.2f%s\n",
                    m, ov, om, bm2, mw, mt, ml, ah, nn, bh, nn, gap,
                    ml > 0 ? "  <-- LOSS" : "");
      }
    }
  }
  return 0;
}

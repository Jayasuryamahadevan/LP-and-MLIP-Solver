#include "ExactBinarySplit.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <thread>

namespace sihps {
namespace {

constexpr int kMaxRows = 8;
constexpr int kMaxBinaries = 62;
using Vec = std::array<std::int64_t, kMaxRows>;

bool is_integral(double v) { return std::fabs(v - std::llround(v)) <= 1e-9; }

struct Extracted {
    int m = 0;                       // equality rows
    int n = 0;                       // binary columns
    std::vector<Vec> col;            // col[j][i] = a_ij, ordered as `binary_col`
    std::vector<std::int32_t> binary_col;  // original column index per j
    std::vector<std::int32_t> slack_col;   // original column index per row
    Vec d{};
};

// Recognizes the structure described in the header. Returns false with a
// reason rather than throwing: an unsupported model is a normal outcome.
bool extract(const MilpProblem& p, Extracted& out, std::string& why) {
    const LpProblem& lp = p.relaxation;
    if (p.maximize) { why = "maximization not supported by this path"; return false; }
    const std::int32_t nr = lp.n_rows(), nc = lp.n_cols();
    if (nr == 0 || nr > kMaxRows) { why = "row count outside supported range"; return false; }
    for (std::int32_t i = 0; i < nr; ++i)
        if (lp.row_types[static_cast<std::size_t>(i)] != 'E') { why = "not all rows are equalities"; return false; }

    // Column-wise view of a row-major matrix.
    std::vector<std::vector<std::pair<std::int32_t, double>>> bycol(static_cast<std::size_t>(nc));
    for (std::int32_t i = 0; i < nr; ++i)
        for (std::int32_t k = lp.A.row_ptr()[i]; k < lp.A.row_ptr()[i + 1]; ++k) {
            const double v = lp.A.values()[static_cast<std::size_t>(k)];
            if (v != 0.0) bycol[static_cast<std::size_t>(lp.A.col_idx()[static_cast<std::size_t>(k)])].push_back({i, v});
        }

    out.m = nr;
    for (std::int32_t i = 0; i < nr; ++i) {
        const double r = lp.rhs[static_cast<std::size_t>(i)];
        if (!is_integral(r)) { why = "non-integral right-hand side"; return false; }
        out.d[static_cast<std::size_t>(i)] = std::llround(r);
    }

    out.slack_col.assign(static_cast<std::size_t>(nr), -1);
    for (std::int32_t j = 0; j < nc; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        const double lo = lp.lower[jj], hi = lp.upper[jj];
        const bool integer = p.variable_types[jj] != VariableType::CONTINUOUS;

        if (integer) {
            if (lo != 0.0 || hi != 1.0) { why = "integer column is not binary"; return false; }
            if (lp.obj[jj] != 0.0) { why = "binary column has a nonzero objective coefficient"; return false; }
            Vec v{};
            for (const auto& [i, a] : bycol[jj]) {
                if (a < 0.0 || !is_integral(a)) { why = "binary coefficient is not a nonnegative integer"; return false; }
                v[static_cast<std::size_t>(i)] = std::llround(a);
            }
            out.col.push_back(v);
            out.binary_col.push_back(j);
        } else {
            if (lo == 0.0 && hi == 0.0) continue;              // inactive slack
            if (bycol[jj].size() != 1) { why = "continuous column spans multiple rows"; return false; }
            const auto& [i, a] = bycol[jj][0];
            if (a != 1.0 || lp.obj[jj] != 1.0 || lo != 0.0) { why = "continuous column is not a unit slack with objective +1"; return false; }
            if (out.slack_col[static_cast<std::size_t>(i)] >= 0) { why = "row carries more than one active slack"; return false; }
            out.slack_col[static_cast<std::size_t>(i)] = j;
        }
    }
    for (std::int32_t i = 0; i < nr; ++i)
        if (out.slack_col[static_cast<std::size_t>(i)] < 0) { why = "a row has no active slack"; return false; }

    out.n = static_cast<int>(out.col.size());
    if (out.n < 2 || out.n > kMaxBinaries) { why = "binary count outside supported range"; return false; }
    return true;
}

int g_m = 0;

std::uint64_t mix(const Vec& v, int m) {
    std::uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < m; ++i) { h ^= static_cast<std::uint64_t>(v[static_cast<std::size_t>(i)]); h *= 1099511628211ULL; }
    h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 32;
    return h;
}

void build_half(std::vector<Vec>& out, const std::vector<Vec>& col, int base, int cnt, int m) {
    out.assign(static_cast<std::size_t>(1) << cnt, Vec{});
    for (std::uint32_t mask = 1; mask < (1u << cnt); ++mask) {
        Vec v = out[mask & (mask - 1)];
        const Vec& c = col[static_cast<std::size_t>(base + __builtin_ctz(mask))];
        for (int i = 0; i < m; ++i) v[static_cast<std::size_t>(i)] += c[static_cast<std::size_t>(i)];
        out[mask] = v;
    }
}

} // namespace

ExactBinarySplitResult try_exact_binary_split(const MilpProblem& problem,
                                              std::uint64_t memory_budget_bytes,
                                              unsigned threads) {
    ExactBinarySplitResult res;
    Extracted ex;
    if (!extract(problem, ex, res.reason)) return res;

    const int m = ex.m, n = ex.n;
    g_m = m;
    if (threads == 0) threads = 1;

    // Smallest columns into L: max(A y_L) is then small, which confines the
    // vector we must look up to a narrow band and keeps the number of
    // (cache-missing) table probes manageable.
    std::vector<int> order(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) order[static_cast<std::size_t>(j)] = j;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        std::int64_t sa = 0, sb = 0;
        for (int i = 0; i < m; ++i) { sa += ex.col[static_cast<std::size_t>(a)][static_cast<std::size_t>(i)];
                                      sb += ex.col[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)]; }
        return sa < sb;
    });
    std::vector<Vec> col(static_cast<std::size_t>(n));
    std::vector<std::int32_t> orig(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        col[static_cast<std::size_t>(j)] = ex.col[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])];
        orig[static_cast<std::size_t>(j)] = ex.binary_col[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])];
    }

    // Largest nL whose table fits the budget, subject to the right half
    // staying enumerable.
    int nL = 0;
    for (int cand = 2; cand <= n - 2 && cand <= 32; ++cand) {
        const std::uint64_t slots = static_cast<std::uint64_t>(2) << cand;
        if (slots * 8ULL <= memory_budget_bytes) nL = cand;
    }
    if (nL < 2) { res.reason = "memory budget too small for any split"; return res; }
    int nR = n - nL;
    if (nR > 34) { res.reason = "right half too large for the memory budget"; return res; }

    const int lLo = nL / 2, lHi = nL - lLo;
    std::vector<Vec> Llo, Lhi;
    build_half(Llo, col, 0, lLo, m);
    build_half(Lhi, col, lLo, lHi, m);

    std::uint64_t slots = static_cast<std::uint64_t>(2) << nL;
    int shift = 64; { std::uint64_t s = slots; while (s > 1) { --shift; s >>= 1; } }
    shift = 64 - shift;
    constexpr std::uint64_t kEmpty = ~0ULL;
    std::vector<std::uint64_t> table;
    try { table.assign(static_cast<std::size_t>(slots), kEmpty); }
    catch (const std::bad_alloc&) { res.reason = "table allocation failed"; return res; }

    auto lvec = [&](std::uint32_t mask) {
        const Vec& a = Llo[mask & ((1u << lLo) - 1u)];
        const Vec& b = Lhi[mask >> lLo];
        Vec r{};
        for (int i = 0; i < m; ++i) r[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)] + b[static_cast<std::size_t>(i)];
        return r;
    };

    Vec maxL{};
    for (int j = 0; j < nL; ++j)
        for (int i = 0; i < m; ++i) maxL[static_cast<std::size_t>(i)] += col[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];

    for (std::uint32_t hi = 0; hi < (1u << lHi); ++hi) {
        const Vec& vh = Lhi[hi];
        bool ok = true;
        for (int i = 0; i < m && ok; ++i) if (vh[static_cast<std::size_t>(i)] > ex.d[static_cast<std::size_t>(i)]) ok = false;
        if (!ok) continue;
        for (std::uint32_t lo = 0; lo < (1u << lLo); ++lo) {
            const Vec& vl = Llo[lo];
            Vec v{}; bool good = true;
            for (int i = 0; i < m; ++i) {
                v[static_cast<std::size_t>(i)] = vl[static_cast<std::size_t>(i)] + vh[static_cast<std::size_t>(i)];
                if (v[static_cast<std::size_t>(i)] > ex.d[static_cast<std::size_t>(i)]) { good = false; break; }
            }
            if (!good) continue;
            const std::uint32_t mask = lo | (hi << lLo);
            const std::uint64_t h = mix(v, m);
            std::size_t idx = static_cast<std::size_t>(h >> shift) & (slots - 1);
            while (table[idx] != kEmpty) idx = (idx + 1) & (slots - 1);
            table[idx] = static_cast<std::uint64_t>(mask) |
                         (static_cast<std::uint64_t>(static_cast<std::uint32_t>(h >> 32)) << 32);
        }
    }

    auto lookup = [&](const Vec& key) -> std::uint32_t {
        const std::uint64_t h = mix(key, m);
        const std::uint32_t tag = static_cast<std::uint32_t>(h >> 32);
        std::size_t idx = static_cast<std::size_t>(h >> shift) & (slots - 1);
        for (;;) {
            const std::uint64_t e = table[idx];
            if (e == kEmpty) return ~0u;
            if (static_cast<std::uint32_t>(e >> 32) == tag) {
                const std::uint32_t mask = static_cast<std::uint32_t>(e & 0xFFFFFFFFu);
                const Vec v = lvec(mask);
                bool eq = true;
                for (int i = 0; i < m && eq; ++i) if (v[static_cast<std::size_t>(i)] != key[static_cast<std::size_t>(i)]) eq = false;
                if (eq) return mask;
            }
            idx = (idx + 1) & (slots - 1);
        }
    };

    const int rLo = nR / 2, rHi = nR - rLo;
    std::vector<Vec> Rlo, Rhi;
    build_half(Rlo, col, nL, rLo, m);
    build_half(Rhi, col, nL + rLo, rHi, m);

    std::int64_t total_d = 0;
    for (int i = 0; i < m; ++i) total_d += ex.d[static_cast<std::size_t>(i)];

    std::atomic<std::uint64_t> subsets{0}, probes{0};

    for (std::int64_t v = 0; v <= total_d; ++v) {
        std::vector<Vec> targets;
        { Vec cur{};
          // compositions of v across m rows
          std::vector<int> stack(static_cast<std::size_t>(m) + 1, 0);
          std::function<void(int, std::int64_t)> gen = [&](int i, std::int64_t left) {
              if (i == m) { if (left == 0) {
                  Vec t{}; bool ok = true;
                  for (int k = 0; k < m; ++k) {
                      t[static_cast<std::size_t>(k)] = ex.d[static_cast<std::size_t>(k)] - cur[static_cast<std::size_t>(k)];
                      if (t[static_cast<std::size_t>(k)] < 0) ok = false;
                  }
                  if (ok) targets.push_back(t);
              } return; }
              for (std::int64_t k = 0; k <= left; ++k) { cur[static_cast<std::size_t>(i)] = k; gen(i + 1, left - k); }
              cur[static_cast<std::size_t>(i)] = 0;
          };
          gen(0, v);
        }
        if (targets.empty()) continue;

        std::atomic<bool> found{false};
        std::atomic<std::uint32_t> f_l{0}, f_r{0};
        std::atomic<std::uint32_t> next_hi{0};
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < threads; ++t) {
            pool.emplace_back([&] {
                for (;;) {
                    const std::uint32_t hi = next_hi.fetch_add(1);
                    if (hi >= (1u << rHi) || found.load(std::memory_order_relaxed)) return;
                    const Vec& vh = Rhi[hi];
                    bool hok = true;
                    for (int i = 0; i < m && hok; ++i) if (vh[static_cast<std::size_t>(i)] > ex.d[static_cast<std::size_t>(i)]) hok = false;
                    if (!hok) continue;
                    for (std::uint32_t lo = 0; lo < (1u << rLo); ++lo) {
                        const Vec& vl = Rlo[lo];
                        Vec vr{}; bool ok = true;
                        for (int i = 0; i < m; ++i) {
                            vr[static_cast<std::size_t>(i)] = vl[static_cast<std::size_t>(i)] + vh[static_cast<std::size_t>(i)];
                            if (vr[static_cast<std::size_t>(i)] > ex.d[static_cast<std::size_t>(i)]) { ok = false; break; }
                        }
                        if (!ok) continue;
                        subsets.fetch_add(1, std::memory_order_relaxed);
                        for (const Vec& tg : targets) {
                            Vec need{}; bool good = true;
                            for (int i = 0; i < m; ++i) {
                                need[static_cast<std::size_t>(i)] = tg[static_cast<std::size_t>(i)] - vr[static_cast<std::size_t>(i)];
                                if (need[static_cast<std::size_t>(i)] < 0 || need[static_cast<std::size_t>(i)] > maxL[static_cast<std::size_t>(i)]) { good = false; break; }
                            }
                            if (!good) continue;
                            probes.fetch_add(1, std::memory_order_relaxed);
                            const std::uint32_t lm = lookup(need);
                            if (lm != ~0u) {
                                bool exp = false;
                                if (found.compare_exchange_strong(exp, true)) { f_l = lm; f_r = lo | (hi << rLo); }
                                return;
                            }
                        }
                    }
                }
            });
        }
        for (auto& th : pool) th.join();

        if (!found.load()) continue;

        // Reconstruct in ORIGINAL column space and re-verify from the
        // original coefficients -- the search result is never trusted on
        // its own.
        res.x.assign(static_cast<std::size_t>(problem.n_cols()), 0.0);
        const std::uint32_t lm = f_l.load(), rm = f_r.load();
        for (int b = 0; b < nL; ++b) if (lm >> b & 1u) res.x[static_cast<std::size_t>(orig[static_cast<std::size_t>(b)])] = 1.0;
        for (int b = 0; b < nR; ++b) if (rm >> b & 1u) res.x[static_cast<std::size_t>(orig[static_cast<std::size_t>(nL + b)])] = 1.0;

        Vec act{};
        for (int j = 0; j < n; ++j) {
            if (res.x[static_cast<std::size_t>(orig[static_cast<std::size_t>(j)])] == 0.0) continue;
            for (int i = 0; i < m; ++i) act[static_cast<std::size_t>(i)] += col[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
        }
        std::int64_t slack_sum = 0;
        for (int i = 0; i < m; ++i) {
            const std::int64_t s = ex.d[static_cast<std::size_t>(i)] - act[static_cast<std::size_t>(i)];
            if (s < 0) { res.reason = "internal: reconstructed point violates a row"; res.x.clear(); return res; }
            res.x[static_cast<std::size_t>(ex.slack_col[static_cast<std::size_t>(i)])] = static_cast<double>(s);
            slack_sum += s;
        }
        if (slack_sum != v) { res.reason = "internal: reconstructed objective disagrees"; res.x.clear(); return res; }

        res.applicable = true;
        res.solved = true;
        res.objective = static_cast<double>(v);
        res.subsets_examined = subsets.load();
        res.probes = probes.load();
        return res;
    }

    res.applicable = true;
    res.reason = "no feasible assignment exists";
    return res;
}

} // namespace sihps

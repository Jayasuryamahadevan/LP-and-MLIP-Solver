// Correctness validation against published Netlib reference optima --
// validation-hierarchy LEVEL 7 (prompt.md): an optimization-model
// benchmark, not merely a numerical-kernel check.
//
// The expected values are NOT hardcoded here. They are parsed at runtime
// from Netlib's own index file (data/netlib_readme.txt, fetched from
// https://www.netlib.org/lp/data/readme) by src/io/NetlibReference.cpp,
// which reads both the PROBLEM SUMMARY TABLE and the later CPLEX/MINOS
// recomputation table. An instance passes if this solver's objective
// matches ANY published value for it, and the report says which one --
// see NetlibReference.hpp for why matching only the primary column would
// produce false failures.
//
// Exit status is nonzero if any instance fails, so this doubles as a
// regression gate.

#include "bench/RunMetadata.hpp"
#include "bench/ResourceSnapshot.hpp"
#include "cuda/CudaDevice.hpp"
#include "io/MpsReader.hpp"
#include "io/NetlibReference.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"
#include "lp/Simplex.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <memory>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace sihps;
namespace fs = std::filesystem;

namespace {

// Relative-objective agreement threshold. Netlib's own primary values
// carry ~10 significant digits, and its CPLEX/MINOS columns disagree with
// each other at the 1e-8..1e-7 level on several instances, so a tighter
// threshold than this would be measuring transcription precision rather
// than solver correctness.
constexpr double kObjTol = 1e-6;

const char* status_str(LpStatus s) {
    switch (s) {
        case LpStatus::OPTIMAL: return "OPTIMAL";
        case LpStatus::INFEASIBLE: return "INFEASIBLE";
        case LpStatus::UNBOUNDED: return "UNBOUNDED";
        case LpStatus::ITERATION_LIMIT: return "ITER_LIMIT";
        case LpStatus::NUMERICAL_FAILURE: return "NUM_FAILURE";
    }
    return "?";
}

} // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "data/netlib_lp/feasible";
    const std::string readme = (argc > 2) ? argv[2] : "data/netlib_readme.txt";
    const std::int32_t max_rows = (argc > 3) ? std::atoi(argv[3]) : 600;
    // argv[4]: "nopresolve" disables presolve, so the two pipelines can be
    // compared on identical instances without rebuilding.
    const bool use_presolve = !(argc > 4 && std::string(argv[4]) == "nopresolve");
    // argv[5]: "hybrid" lets the first-order solver pick up any instance the
    // simplex abandons. Off by default so the simplex's own pass rate stays
    // measurable rather than being quietly propped up by the fallback.
    const bool hybrid = (argc > 5 && std::string(argv[5]) == "hybrid");
    // argv[6]: path for the JSON Lines record. Every quoted number in this
    // project's documentation should be traceable to one of these files.
    const std::string jsonl_path = (argc > 6) ? argv[6] : "";
    // argv[7]: how many times to solve EACH instance. Default 1 preserves
    // every prior measurement's exact meaning (docs/measurements/*.jsonl
    // predate this option and are all implicitly repeat_count=1); above 1,
    // InstanceRecord.wall_seconds becomes the MEDIAN of that many solves
    // (bench::Summary's own median convention: sorted[n/2]) with
    // wall_seconds_min/max recorded alongside, so a later regression check
    // can tell real change from run-to-run noise (docs/ROADMAP_STATUS.md
    // Phase 0). Repeating also re-verifies this solver's stated
    // determinism-under-fixed-configuration design goal for free: if two
    // solves of the identical instance disagree on objective or iteration
    // count, that is flagged, not averaged away.
    const int repeats = (argc > 7) ? std::max(1, std::atoi(argv[7])) : 1;
    // argv[8]: "doubleton" enables LpSolverOptions::enable_doubleton_
    // substitution (off by default, matching that option's own default).
    const bool doubleton = argc > 8 && std::string(argv[8]) == "doubleton";
    // SIHPS_PDLP_ADAPTIVE=0 disables the adaptive step size for the whole
    // sweep. An environment variable rather than another positional
    // argument because this exists to A/B one default, not to be a
    // permanent part of the interface.
    bool pdlp_adaptive = true;
    if (const char* e = std::getenv("SIHPS_PDLP_ADAPTIVE")) pdlp_adaptive = (std::atoi(e) != 0);

    const bench::ResourceSnapshot resources_before = bench::capture_resources();
    const auto benchmark_start = std::chrono::steady_clock::now();

    const bench::RunMetadata meta = bench::RunMetadata::capture();
    bench::RunConfig cfg;
    cfg.method = hybrid ? "HYBRID" : "SIMPLEX";
    cfg.pricing_rule = "DEVEX";
    cfg.algorithm = "AUTO";
    cfg.presolve = use_presolve;
    cfg.ruiz_scaling = true;
    cfg.row_cap = max_rows;
    {
        LpSolverOptions defaults;
        cfg.hybrid_simplex_budget_seconds = defaults.hybrid_simplex_budget_seconds;
        cfg.hybrid_first_order_eps = defaults.hybrid_first_order_eps;
    }

    std::printf("build %s (%s)  %s  CUDA %s  %s\n", meta.git_commit.c_str(),
                meta.git_dirty.c_str(), meta.compiler.c_str(), meta.cuda_version.c_str(),
                meta.gpu_name.c_str());
    std::printf("threads=%d  method=%s  presolve=%s  pdlp_adaptive=%s  doubleton=%s\n\n",
                meta.thread_count, cfg.method.c_str(), cfg.presolve ? "on" : "off",
                pdlp_adaptive ? "on" : "off", doubleton ? "on" : "off");

    std::unique_ptr<bench::JsonlWriter> writer;
    if (!jsonl_path.empty()) {
        writer.reset(new bench::JsonlWriter(jsonl_path, meta, cfg));
        if (!writer->ok()) {
            std::printf("WARNING: could not open %s for writing; continuing without records\n",
                        jsonl_path.c_str());
            writer.reset();
        }
    }
    std::vector<bench::InstanceRecord> records;

    NetlibReference ref;
    try {
        ref = NetlibReference::load(readme);
    } catch (const std::exception& e) {
        std::printf("Cannot load reference values: %s\n", e.what());
        return 2;
    }
    std::printf("Loaded published reference values for %zu instances from %s\n\n", ref.size(),
                readme.c_str());

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".mps") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    std::printf("%14s %6s %6s %11s %20s %20s %11s %8s %8s %5s %s\n", "instance", "rows", "cols",
                "status", "our objective", "reference", "rel.err", "sec", "iters", "refac",
                "verdict");
    std::printf("%s\n", std::string(146, '-').c_str());

    int passed = 0, failed = 0, no_ref = 0, skipped = 0;
    std::vector<std::string> failures;

    for (const auto& path : files) {
        const std::string name = path.stem().string();

        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception&) {
            ++skipped;
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0 || model.n_rows > max_rows) {
            ++skipped;
            continue;
        }

        const auto* refs = ref.find(name);
        if (!refs || refs->empty()) {
            ++no_ref;
            continue;
        }

        LpProblem p = lp_problem_from_mps(model);
        LpSolverOptions options;
        options.use_presolve = use_presolve;
        if (hybrid) options.method = LpMethod::HYBRID;
        options.pdlp.adaptive_step = pdlp_adaptive;
        options.enable_doubleton_substitution = doubleton;

        LpSolution result;
        std::vector<double> rep_seconds;
        rep_seconds.reserve(static_cast<std::size_t>(repeats));
        bool deterministic = true;
        for (int rep = 0; rep < repeats; ++rep) {
            LpSolution this_result;
            const auto t0 = std::chrono::steady_clock::now();
            try {
                this_result = solve_lp(p, options);
            } catch (const std::exception&) {
                this_result.status = LpStatus::NUMERICAL_FAILURE;
            }
            rep_seconds.push_back(
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            if (rep == 0) {
                result = this_result;
            } else if (this_result.status != result.status ||
                       this_result.iterations != result.iterations ||
                       this_result.objective_value != result.objective_value) {
                deterministic = false;
            }
        }
        std::sort(rep_seconds.begin(), rep_seconds.end());
        const double secs = rep_seconds[rep_seconds.size() / 2]; // median; single sample when repeats==1
        const double secs_min = rep_seconds.front();
        const double secs_max = rep_seconds.back();
        if (!deterministic) {
            std::printf("  WARNING: %s did not reproduce identically across %d repeats "
                        "(status/iterations/objective differed)\n",
                        name.c_str(), repeats);
        }

        // Compare against the closest published value; report which one.
        double best_rel = 1e300;
        std::string best_source;
        double best_ref = 0.0;
        for (const auto& rv : *refs) {
            const double rel =
                std::fabs(result.objective_value - rv.value) / (1.0 + std::fabs(rv.value));
            if (rel < best_rel) {
                best_rel = rel;
                best_source = rv.source;
                best_ref = rv.value;
            }
        }

        const bool ok = (result.status == LpStatus::OPTIMAL) && (best_rel < kObjTol);
        if (ok) {
            ++passed;
        } else {
            ++failed;
            failures.push_back(name);
        }

        std::printf("%14s %6d %6d %11s %20.8f %20.8f %11.2e %8.3f %8d %5d %s (%s)%s\n",
                    name.c_str(), model.n_rows, model.n_cols, status_str(result.status),
                    result.objective_value, best_ref, best_rel, secs, result.iterations,
                    result.refactorizations, ok ? "PASS" : "FAIL", best_source.c_str(),
                    result.used_first_order ? " [first-order]" : "");
        if (repeats > 1) {
            std::printf("               (%d repeats: median %.3f s, min %.3f s, max %.3f s)\n",
                        repeats, secs, secs_min, secs_max);
        }

        const bench::ResourceSnapshot after_instance = bench::capture_resources();

        bench::InstanceRecord rec;
        rec.instance_path = path.string();
        rec.instance_hash = bench::hash_file(path.string());
        rec.rows = model.n_rows;
        rec.cols = model.n_cols;
        rec.nnz = p.A.nnz();
        rec.status = status_str(result.status);
        rec.objective = result.objective_value;
        rec.reference_objective = best_ref;
        rec.relative_objective_error = best_rel;
        rec.passed = ok;
        rec.reference_source = best_source;
        rec.wall_seconds = secs;
        rec.wall_seconds_min = secs_min;
        rec.wall_seconds_max = secs_max;
        rec.repeat_count = repeats;
        rec.repeats_deterministic = deterministic;
        rec.presolve_seconds = result.presolve_seconds;
        rec.solve_seconds = result.solve_seconds;
        rec.iterations = result.iterations;
        rec.refactorizations = result.refactorizations;
        rec.primal_residual = result.primal_residual;
        rec.dual_residual = result.dual_residual;
        rec.used_first_order = result.used_first_order;
        rec.first_order_fallback_used = result.first_order_fallback_used;
        rec.pdlp_iterations = result.pdlp.iterations;
        rec.pdlp_host_syncs = result.pdlp.host_syncs;
        rec.presolve_removed_rows = result.presolve_removed_rows;
        rec.presolve_removed_cols = result.presolve_removed_cols;
        rec.peak_rss_kb = after_instance.peak_rss_kb;
        rec.gpu_available = after_instance.gpu_available;
        if (after_instance.gpu_available) {
            rec.gpu_used_mb = static_cast<double>(after_instance.gpu_total_bytes -
                                                   after_instance.gpu_free_bytes) /
                               (1024.0 * 1024.0);
        }
        records.push_back(rec);
        if (writer) writer->write(rec);
    }

    std::printf("\n%s\n", std::string(128, '=').c_str());
    std::printf("PASSED %d / %d validated instances (%d had no published reference, %d skipped by "
                "row cap %d)\n",
                passed, passed + failed, no_ref, skipped, max_rows);
    if (!failures.empty()) {
        std::printf("FAILED:");
        for (const auto& f : failures) std::printf(" %s", f.c_str());
        std::printf("\n");
    }

    // The roadmap's LP performance gate: an aggregate that a mean alone
    // would misrepresent, since solve times here span four orders of
    // magnitude and a single slow instance dominates any arithmetic mean.
    const bench::Summary sum = bench::summarize(records);
    std::printf("\nsolved %d / %d   total %.3f s   geomean %.3f s   median %.3f s   "
                "p95 %.3f s   max %.3f s\n",
                sum.solved, sum.attempted, sum.total_seconds, sum.geometric_mean_seconds,
                sum.median_seconds, sum.p95_seconds, sum.max_seconds);
    std::printf("total iterations %lld   worst relative objective error %.3e\n",
                static_cast<long long>(sum.total_iterations), sum.worst_relative_objective_error);
    const bench::ResourceSnapshot resources_after = bench::capture_resources();
    const double wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start).count();
    const double cpu_seconds =
        std::max(0.0, resources_after.cpu_seconds - resources_before.cpu_seconds);
    const double cpu_utilization = wall_seconds > 1e-9 ? 100.0 * cpu_seconds / wall_seconds : 0.0;
    std::printf("resources: wall %.3f s  CPU %.3f s  aggregate CPU utilization %.1f%%  "
                "peak RSS %.1f MB\n",
                wall_seconds, cpu_seconds, cpu_utilization,
                resources_after.peak_rss_kb / 1024.0);
    if (resources_before.gpu_available && resources_after.gpu_available) {
        std::printf("GPU memory: %.1f MB used before -> %.1f MB used after (of %.1f MB)\n",
                    (resources_before.gpu_total_bytes - resources_before.gpu_free_bytes) /
                        (1024.0 * 1024.0),
                    (resources_after.gpu_total_bytes - resources_after.gpu_free_bytes) /
                        (1024.0 * 1024.0),
                    resources_after.gpu_total_bytes / (1024.0 * 1024.0));
    } else {
        std::printf("GPU memory: unavailable\n");
    }
    if (writer) std::printf("records written to %s\n", jsonl_path.c_str());
    return failed == 0 ? 0 : 1;
}

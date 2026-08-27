#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sihps {
namespace bench {

// ---------------------------------------------------------------------
// Reproducibility metadata for every benchmark record.
// ---------------------------------------------------------------------
//
// WHY THIS EXISTS
// ---------------
// CLAUDE_OPUS_SOLVER_ROADMAP.md makes benchmark and observability
// infrastructure priority 1, ahead of every algorithmic change. That
// ordering is not bureaucratic here -- this project has already drawn
// three wrong conclusions from unlabelled measurements:
//
//   1. "Devex has a 26x per-iteration defect on stocfor3."  It did not.
//      Two benchmark processes were oversubscribing 16 cores.
//   2. "GPU PDLP beats the CPU simplex 39x on degen3."  It does not. Same
//      cause; degen3 is a tie and d2q06c reverses to a 2.8x loss.
//   3. "dfl001 is unsolvable by the first-order path."  It is not. The
//      tolerance was set tighter than the validator required.
//
// Every one of those would have been caught immediately had the run
// recorded what else was executing, which binary produced it, and which
// options were in force. A number without its conditions is not a
// measurement, and this header is what turns one into the other.
//
// The fields follow the roadmap's Phase 0 list. Anything genuinely
// unavailable on a platform is reported as "unknown" rather than omitted,
// so a reader can tell a missing field from an unrecorded one.

struct RunMetadata {
    std::string git_commit;      // HEAD at build time, or "unknown"
    std::string git_dirty;       // "clean" | "dirty" | "unknown"
    std::string compiler;        // e.g. "GNU 13.3.0"
    std::string cuda_version;    // toolkit version this was compiled against
    std::string gpu_name;
    std::string gpu_compute_capability;
    std::string gpu_driver;
    std::string cpu_model;
    std::string host_ram;
    std::int32_t thread_count = 0;     // OpenMP threads actually available
    std::string openmp_schedule;
    std::string build_type;

    // Collects everything obtainable on this host. Cheap enough to call
    // once per benchmark process; never call it inside a timed region.
    static RunMetadata capture();
};

// One benchmarked instance. Fields the roadmap's LP KPI list requires;
// anything not applicable to a given run stays at its default so that a
// consumer can distinguish "zero" from "not measured" via `status`.
struct InstanceRecord {
    std::string instance_path;
    std::string instance_hash;   // FNV-1a 64 of the file bytes
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::int32_t nnz = 0;

    std::string status;
    double objective = 0.0;
    double reference_objective = 0.0;
    double relative_objective_error = 0.0;
    bool passed = false;
    std::string reference_source;

    // wall_seconds is the MEDIAN over repeat_count solves (identical to
    // the single sample when repeat_count == 1, so every consumer written
    // before repeated runs existed keeps reading exactly what it always
    // read). wall_seconds_min/max bound the same sample set -- needed so
    // a future regression check can tell real change from run-to-run
    // noise, per docs/ROADMAP_STATUS.md's Phase 0 gap.
    double wall_seconds = 0.0;
    double wall_seconds_min = 0.0;
    double wall_seconds_max = 0.0;
    std::int32_t repeat_count = 1;
    // False if repeated solves of the SAME instance under the SAME
    // configuration produced a different objective or iteration count --
    // this solver's stated design goal is determinism under fixed
    // configuration (docs/ROADMAP_STATUS.md Phase 1), so this is a
    // correctness signal, not just a benchmarking curiosity. Meaningless
    // (left true) when repeat_count == 1.
    bool repeats_deterministic = true;
    double presolve_seconds = 0.0;
    double solve_seconds = 0.0;
    std::int32_t iterations = 0;
    std::int32_t refactorizations = 0;
    double primal_residual = 0.0;
    double dual_residual = 0.0;

    // Cross-method runs solve the SAME instance twice. Recording only a
    // combined wall time loses the one quantity such a run exists to
    // produce -- which method was faster, and by how much -- so both are
    // kept. Zero means that method was not run.
    double simplex_seconds = 0.0;
    double first_order_seconds = 0.0;

    // Which path actually produced the answer. The hybrid engine can
    // return a first-order result, and a benchmark that hides that cannot
    // be used to reason about either method.
    bool used_first_order = false;
    bool first_order_fallback_used = false;
    std::int32_t pdlp_iterations = 0;
    std::int32_t pdlp_host_syncs = 0;

    std::int32_t presolve_removed_rows = 0;
    std::int32_t presolve_removed_cols = 0;

    // Resource usage captured once, after this instance finishes (see
    // ResourceSnapshot.hpp for what "peak" means inside a multi-instance
    // sweep: the process's cumulative peak through this point, not an
    // isolated per-instance figure). Zero/false when not captured.
    long peak_rss_kb = 0;
    bool gpu_available = false;
    double gpu_used_mb = 0.0;
};

// Solver configuration in force for a whole sweep.
struct RunConfig {
    std::string method;
    std::string pricing_rule;
    std::string algorithm;
    bool presolve = true;
    bool ruiz_scaling = true;
    double hybrid_simplex_budget_seconds = 0.0;
    double hybrid_first_order_eps = 0.0;
    std::int32_t row_cap = 0;
};

// FNV-1a 64 over a file's bytes, rendered as 16 lowercase hex digits.
// Returns "unreadable" if the file cannot be opened -- a benchmark should
// still record the attempt rather than silently dropping the instance.
std::string hash_file(const std::string& path);

// Writes one JSON Lines file: a single header object carrying metadata and
// configuration, then one object per instance. JSONL rather than a single
// JSON document so that a killed sweep still leaves valid, parseable
// records for everything it finished.
class JsonlWriter {
public:
    JsonlWriter(const std::string& path, const RunMetadata& meta, const RunConfig& config);
    ~JsonlWriter();

    JsonlWriter(const JsonlWriter&) = delete;
    JsonlWriter& operator=(const JsonlWriter&) = delete;

    void write(const InstanceRecord& record);
    bool ok() const { return file_ != nullptr; }

private:
    void* file_ = nullptr; // std::FILE*, kept opaque to avoid <cstdio> here
};

// Aggregate KPIs the roadmap's LP performance gate requires. Computed over
// the instances a sweep actually solved.
struct Summary {
    std::int32_t attempted = 0;
    std::int32_t solved = 0;
    double total_seconds = 0.0;
    double geometric_mean_seconds = 0.0;
    double median_seconds = 0.0;
    double p95_seconds = 0.0;
    double max_seconds = 0.0;
    std::int64_t total_iterations = 0;
    double worst_relative_objective_error = 0.0;
};

Summary summarize(const std::vector<InstanceRecord>& records);

} // namespace bench
} // namespace sihps

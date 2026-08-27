#pragma once

#include <cstddef>

namespace sihps {
namespace bench {

// Process-wide resource usage snapshot: CPU time, peak RSS, and current
// GPU memory occupancy. Factored out here because bench_miplib.cpp and
// validate_netlib.cpp each carried an identical copy of this before
// (docs/ROADMAP_STATUS.md Phase 0: "reuse ... don't reinvent" applies to
// this project's own code, not only to external libraries).
//
// peak_rss_kb comes from getrusage's ru_maxrss, which is the peak
// resident set size for the WHOLE PROCESS since it started, monotonic
// non-decreasing -- it is not reset between calls. A snapshot taken after
// each instance in a multi-instance sweep therefore reports "the
// process's peak so far, through this instance," not an isolated
// per-instance figure; by the final instance it equals the sweep's true
// peak. This is a real limitation, stated rather than hidden, and it is
// the same interpretation bench_miplib.cpp's per-instance RSS column
// already carried before this file existed.
struct ResourceSnapshot {
    double cpu_seconds = 0.0;
    long peak_rss_kb = 0;
    bool gpu_available = false;
    std::size_t gpu_free_bytes = 0;
    std::size_t gpu_total_bytes = 0;
};

// Cheap enough to call around any region (a getrusage syscall plus, if a
// GPU is present, a CUDA memory-info query); never call inside a timed
// hot loop.
ResourceSnapshot capture_resources();

} // namespace bench
} // namespace sihps

#include "test_framework.hpp"

#include "bench/ResourceSnapshot.hpp"

#include <vector>

using sihps::bench::capture_resources;

SIHPS_TEST(resource_snapshot_reports_nonzero_peak_rss) {
    const auto snap = capture_resources();
    // A process that has loaded and run a test binary has resident
    // memory; zero would mean getrusage silently failed rather than that
    // the process is empty.
    SIHPS_ASSERT_TRUE(snap.peak_rss_kb > 0);
}

SIHPS_TEST(resource_snapshot_peak_rss_is_monotonic_non_decreasing) {
    // ru_maxrss (ResourceSnapshot.hpp's documented contract) is the peak
    // for the WHOLE PROCESS since it started -- a second snapshot after
    // allocating and touching a sizeable block must never report a
    // smaller peak than the first.
    const auto before = capture_resources();
    std::vector<double> churn(8 * 1024 * 1024, 1.0); // ~64 MB, forces real pages
    churn[0] = churn[churn.size() - 1]; // touch it so it cannot be elided
    const auto after = capture_resources();
    SIHPS_ASSERT_TRUE(after.peak_rss_kb >= before.peak_rss_kb);
    SIHPS_ASSERT_TRUE(churn[0] == 1.0); // keep the allocation live for the compiler
}

SIHPS_TEST(resource_snapshot_gpu_fields_are_internally_consistent) {
    const auto snap = capture_resources();
    if (snap.gpu_available) {
        SIHPS_ASSERT_TRUE(snap.gpu_total_bytes > 0);
        SIHPS_ASSERT_TRUE(snap.gpu_free_bytes <= snap.gpu_total_bytes);
    }
}

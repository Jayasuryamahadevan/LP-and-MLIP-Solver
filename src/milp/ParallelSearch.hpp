#pragma once

#include "../lp/LpProblem.hpp"
#include "../lp/Simplex.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace sihps {

// Concurrency primitives for parallel branch-and-bound (src/milp/
// MilpSolver.cpp). Deliberately simple -- a single mutex-guarded priority
// queue and a fixed worker pool, not per-thread work-stealing deques or
// lock-free structures -- matching this project's own established
// practice of shipping the simplest correct thing first and gating
// anything more elaborate behind a measured bottleneck
// (docs/architecture/MILP.md's parallel-B&B section has the full design
// rationale, including why a single SHARED queue -- preserving global
// best-bound node ordering across every worker -- was chosen over
// per-thread queues, which would each maintain an independent frontier
// and could waste work exploring nodes far from the true global bound).

// Thread-safe best-bound priority queue. T and Compare mirror
// std::priority_queue's own template parameters; in this project's actual
// use, T is `std::shared_ptr<const SearchNode>` and Compare is the same
// NodeCompare the pre-existing single-threaded queue already used, so
// parallel search preserves identical node ORDERING PREFERENCE to the
// serial code -- only the interleaving across workers is new.
template <typename T, typename Compare>
class ConcurrentPriorityQueue {
public:
    ConcurrentPriorityQueue() = default;

    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(value));
        }
        not_empty_.notify_one();
    }

    // Pushes two values (a B&B node's two children) under a single lock
    // acquisition -- every non-leaf node in the whole search pushes
    // exactly two children, so this is the hottest push path and is worth
    // not paying the lock/unlock cost twice for.
    void push_pair(T first, T second) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(first));
            queue_.push(std::move(second));
        }
        not_empty_.notify_all();
    }

    // Blocking pop against a WorkerCoordinator implementing the classic
    // "every worker idle AND queue empty" termination check. Returns
    // std::nullopt only once the coordinator has declared termination --
    // NEVER merely because the queue looked empty at some instant, since
    // a sibling worker may be a single instruction away from pushing new
    // children. idle_count/terminated below are intentionally guarded by
    // THIS queue's own mutex_ rather than a separate lock: that is what
    // makes "queue empty AND all idle" an atomic, race-free observation
    // (see the WorkerCoordinator's own comment for the textbook race this
    // avoids).
    template <typename Coordinator>
    std::optional<T> pop_or_wait(Coordinator& coord) {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            if (!queue_.empty()) {
                T value = std::move(const_cast<T&>(queue_.top()));
                queue_.pop();
                return value;
            }
            if (coord.terminated || coord.external_stop.load(std::memory_order_relaxed)) {
                coord.terminated = true;
                not_empty_.notify_all();
                return std::nullopt;
            }
            ++coord.idle_count;
            if (coord.idle_count >= coord.n_workers) {
                coord.terminated = true;
                not_empty_.notify_all();
                return std::nullopt;
            }
            // Bounded wait, not an unbounded one: a defensive poll
            // fallback so a worker can never sleep indefinitely if some
            // future code path forgets to call request_stop()'s own
            // notify -- cheap at this interval, and this is a genuine
            // engineering-margin choice (ENGINEERING TECHNIQUE), not a
            // claim that the notify-based path alone is unreliable.
            not_empty_.wait_for(lock, std::chrono::milliseconds(50));
            --coord.idle_count;
        }
    }

    // External stop request (a fatal error or a global time/node limit
    // observed by one worker): sets the flag AND wakes every worker
    // currently blocked in pop_or_wait so they notice promptly.
    template <typename Coordinator>
    void request_stop(Coordinator& coord) {
        coord.external_stop.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex_);
        not_empty_.notify_all();
    }

    std::size_t size_unsafe() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // The single best (lowest, for a minimization LP bound) priority_bound
    // still open, or nullopt if empty -- used for the final best_bound/gap
    // report exactly as the serial code's current_best_bound() already
    // does, just made safe to call after every worker has joined (at that
    // point it's uncontended, but the lock is still correct and cheap).
    template <typename BoundOf>
    std::optional<double> best_priority_bound(BoundOf bound_of) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        return bound_of(queue_.top());
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::priority_queue<T, std::vector<T>, Compare> queue_;
};

// Termination detector for a fixed-size worker pool pulling from one
// ConcurrentPriorityQueue. idle_count and terminated are read/written
// ONLY from inside ConcurrentPriorityQueue::pop_or_wait, which holds that
// queue's own mutex for every access -- sharing that lock (rather than
// giving this class its own) is what prevents the textbook race where a
// worker observes "queue empty, every OTHER worker idle" and declares
// termination the same instant a sibling is one instruction away from
// pushing new children: since pushing also requires that same mutex, no
// push can be "in flight" while this check runs.
struct WorkerCoordinator {
    explicit WorkerCoordinator(unsigned workers) : n_workers(workers) {}

    const unsigned n_workers;
    unsigned idle_count = 0;              // GUARDED by the queue's mutex_, not a separate lock
    bool terminated = false;              // GUARDED by the queue's mutex_
    std::atomic<bool> external_stop{false};
};

// Global, cross-worker incumbent. Deliberately split into two pieces with
// DIFFERENT synchronization strategies, not one:
//
//  - `objective` is a lock-free atomic, read at every single prune check
//    in every worker (the hottest read in the whole search) via
//    load(memory_order_acquire) -- this MUST NOT take a lock.
//  - `x` (and the bookkeeping counter) are guarded by `mutex`, updated
//    only from inside the serialized compare-and-update below. A pure
//    lock-free CAS on `objective` alone would let a reader observe a NEW
//    objective value paired with the OLD `x` -- a torn update -- so the
//    pair is updated together under one lock, not independently.
struct IncumbentState {
    std::atomic<double> objective{std::numeric_limits<double>::infinity()};
    std::mutex mutex;
    std::vector<double> x;                // GUARDED by mutex
    std::uint64_t incumbent_updates = 0;  // GUARDED by mutex

    // Lock-free fast rejection: true if `candidate_objective` is clearly
    // not an improvement, checked WITHOUT taking `mutex`. This is what
    // keeps the common case (the rounding heuristic firing every node,
    // usually not improving) off the lock entirely. A caller that gets
    // `false` here must still re-check under the lock before writing --
    // see MilpSolver.cpp's consider_incumbent_mt for the exact protocol.
    bool clearly_worse(double candidate_objective, double objective_tolerance) const {
        const double current = objective.load(std::memory_order_acquire);
        return std::isfinite(current) &&
               candidate_objective >= current - objective_tolerance * (1.0 + std::fabs(current));
    }
};

// Per-worker state: everything a worker thread touches that must NOT be
// shared with any other worker. Given each of this project's own MILP
// benchmark instances has small per-node LP relaxations (7-164 rows), the
// per-worker LpProblem copy below is a small, fixed cost paid once per
// worker at startup, not per node.
struct WorkerContext {
    explicit WorkerContext(LpProblem workspace_in, std::size_t n_cols)
        : workspace(std::move(workspace_in)),
          down_pseudocost(n_cols, 0.0),
          up_pseudocost(n_cols, 0.0),
          down_observations(n_cols, 0),
          up_observations(n_cols, 0) {}

    LpProblem workspace;
    std::vector<double> down_pseudocost, up_pseudocost;
    std::vector<std::uint32_t> down_observations, up_observations;
    // Keyed by SearchNode::order (globally unique, assigned under the
    // queue's own mutex when children are created -- see MilpSolver.cpp).
    // A node is only ever popped by ONE worker, so only that worker ever
    // needs its own parent's basis entry; a miss here (e.g. because a
    // node's children got reassigned to a DIFFERENT worker than the one
    // that created them, which the shared queue permits) is a pure
    // warm-start-hit-rate degradation, never a correctness issue -- the
    // existing single-threaded code already treats a pending_basis miss
    // as "fall back to a cold solve," unchanged here.
    std::unordered_map<std::uint64_t, std::shared_ptr<const Simplex::Basis>> pending_basis;

    // Per-worker counters, summed into the real MilpSolution once every
    // worker has joined -- strictly better than a shared atomic per
    // counter (no false sharing, no per-increment atomic RMW cost on a
    // counter incremented every single node).
    std::uint64_t nodes_processed = 0;
    std::uint64_t nodes_pruned = 0;
    std::uint64_t lp_relaxations = 0;
    std::uint64_t strong_branching_probes = 0;
    std::uint64_t warm_started_relaxations = 0;
    std::uint64_t warm_start_verification_fallbacks = 0;
    std::uint64_t diving_heuristic_lp_relaxations = 0;
    std::uint64_t local_improvement_lp_relaxations = 0;
    std::uint64_t rens_heuristic_lp_relaxations = 0;
    // Root-only in this codebase (cut separation self-gates on
    // node->depth == 0), so only ever nonzero on the sequential root
    // context -- still tracked per-WorkerContext, not a separate
    // mechanism, so MilpSolver.cpp sums it the same way as every other
    // counter here.
    std::uint64_t root_cover_cuts = 0;
    std::uint64_t root_gmi_cuts = 0;
};

} // namespace sihps

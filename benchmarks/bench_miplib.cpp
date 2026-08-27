#include "bench/ResourceSnapshot.hpp"
#include "io/MpsReader.hpp"
#include "milp/MilpProblem.hpp"
#include "milp/MilpSolver.hpp"
#include "cuda/CudaDevice.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Reference {
    std::string status;
    double objective = 0.0;
};

using ResourceSnapshot = sihps::bench::ResourceSnapshot;

ResourceSnapshot resources() { return sihps::bench::capture_resources(); }

std::unordered_map<std::string, Reference> read_solution_file(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open solution file: " + path.string());

    std::unordered_map<std::string, Reference> references;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string marker;
        std::string name;
        if (!(row >> marker >> name)) continue;
        if (marker == "=inf=") {
            references[name] = {"INFEASIBLE", 0.0};
        } else if (marker == "=opt=" || marker == "=best=") {
            double objective = 0.0;
            if (row >> objective) references[name] = {marker == "=opt=" ? "OPTIMAL" : "BEST", objective};
        }
    }
    return references;
}

const char* status_name(sihps::MilpStatus status) {
    switch (status) {
    case sihps::MilpStatus::OPTIMAL: return "OPTIMAL";
    case sihps::MilpStatus::INFEASIBLE: return "INFEASIBLE";
    case sihps::MilpStatus::UNBOUNDED: return "UNBOUNDED";
    case sihps::MilpStatus::UNBOUNDED_RELAXATION: return "UNBOUNDED_RELAXATION";
    case sihps::MilpStatus::NODE_LIMIT: return "NODE_LIMIT";
    case sihps::MilpStatus::TIME_LIMIT: return "TIME_LIMIT";
    case sihps::MilpStatus::NUMERICAL_FAILURE: return "NUMERICAL_FAILURE";
    }
    return "UNKNOWN";
}

} // namespace

int main(int argc, char** argv) {
    const fs::path instance_dir = argc > 1 ? argv[1] : "data/miplib2017_small";
    const fs::path solution_path = argc > 2 ? argv[2] : instance_dir / "miplib2017-v36.solu";
    const std::string selected_instance = argc > 3 ? argv[3] : "";
    const double time_limit = argc > 4 ? std::stod(argv[4]) : 60.0;
    const std::string branching_rule = argc > 5 ? argv[5] : "reliability";
    // docs/architecture/LP.md \S1/\S2, MilpSolverOptions::
    // warm_start_node_relaxations: off by default, matching that option's
    // own default, so a plain invocation of this benchmark reproduces the
    // existing baseline exactly.
    const bool warm_start = argc > 6 && std::string(argv[6]) == "on";
    // MilpSolverOptions::enable_root_gmi_cuts: off by default, matching
    // that option's own default (docs/architecture/MILP.md \S3), so a
    // plain invocation reproduces the existing baseline exactly.
    const bool gmi_cuts = argc > 7 && std::string(argv[7]) == "on";
    // MilpSolverOptions::enable_integer_bound_rounding: off by default,
    // matching that option's own default, so a plain invocation reproduces
    // the existing baseline exactly.
    const bool integer_bound_rounding = argc > 8 && std::string(argv[8]) == "on";

    std::cout << std::unitbuf;

    const auto references = read_solution_file(solution_path);
    std::vector<fs::path> instances;
    for (const auto& entry : fs::directory_iterator(instance_dir)) {
        if (entry.path().extension() == ".mps" &&
            (selected_instance.empty() || entry.path().stem().string() == selected_instance)) {
            instances.push_back(entry.path());
        }
    }
    std::sort(instances.begin(), instances.end());

    std::size_t exact_matches = 0;
    std::size_t incumbent_matches = 0;
    std::size_t certified_results = 0;
    std::size_t mismatches = 0;
    std::size_t limits = 0;
    constexpr double kObjectiveTolerance = 1e-5;

    const ResourceSnapshot process_start = resources();
    std::cout << "time_limit_seconds=" << time_limit << " branching_rule=" << branching_rule
              << " warm_start=" << (warm_start ? "on" : "off")
              << " gmi_cuts=" << (gmi_cuts ? "on" : "off")
              << " integer_bound_rounding=" << (integer_bound_rounding ? "on" : "off")
              << " gpu_available=" << (process_start.gpu_available ? "yes" : "no") << '\n';
    std::cout << std::left << std::setw(18) << "instance" << std::right << std::setw(12)
              << "status" << std::setw(18) << "ours" << std::setw(18) << "reference"
              << std::setw(12) << "abs_error" << std::setw(10) << "nodes" << std::setw(10)
              << "LPs" << std::setw(10) << "seconds" << std::setw(10) << "cpu_s"
              << std::setw(9) << "CPU%" << std::setw(12) << "RSS_MB" << std::setw(12)
              << "GPU_MB" << std::setw(8) << "cuts" << std::setw(8) << "gmi"
              << std::setw(12) << "best_bound"
              << std::setw(10) << "gap" << std::setw(10) << "warm" << std::setw(10)
              << "warm_fb"
              << "  verdict\n";
    std::cout << std::string(203, '-') << '\n';

    for (const fs::path& instance : instances) {
        const std::string name = instance.stem().string();
        const auto reference_it = references.find(name);
        if (reference_it == references.end()) {
            std::cout << name << " missing reference\n";
            ++mismatches;
            continue;
        }

        try {
            const auto model = sihps::read_mps_file(instance.string());
            const auto problem = sihps::milp_problem_from_mps(model);
            sihps::MilpSolverOptions options;
            options.time_limit_seconds = time_limit;
            options.use_rounding_heuristic = true;
            options.warm_start_node_relaxations = warm_start;
            options.enable_root_gmi_cuts = gmi_cuts;
            options.enable_integer_bound_rounding = integer_bound_rounding;
            if (branching_rule == "most") {
                options.branching_rule = sihps::MilpBranchingRule::MOST_FRACTIONAL;
            } else if (branching_rule == "pseudocost") {
                options.branching_rule = sihps::MilpBranchingRule::PSEUDOCOST;
            } else if (branching_rule != "reliability") {
                throw std::invalid_argument("branching rule must be reliability, pseudocost, or most");
            }

            const ResourceSnapshot before = resources();
            const auto start = std::chrono::steady_clock::now();
            const auto result = sihps::solve_milp(problem, options);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const ResourceSnapshot after = resources();
            const double cpu_seconds = std::max(0.0, after.cpu_seconds - before.cpu_seconds);
            const double cpu_utilization = seconds > 1e-9 ? 100.0 * cpu_seconds / seconds : 0.0;
            const double rss_mb = static_cast<double>(after.peak_rss_kb) / 1024.0;
            const double gpu_before_mb = before.gpu_available
                                             ? static_cast<double>(before.gpu_total_bytes -
                                                                   before.gpu_free_bytes) /
                                                   (1024.0 * 1024.0)
                                             : 0.0;
            const double gpu_after_mb = after.gpu_available
                                            ? static_cast<double>(after.gpu_total_bytes -
                                                                  after.gpu_free_bytes) /
                                                  (1024.0 * 1024.0)
                                            : 0.0;

            const Reference& reference = reference_it->second;
            const bool status_match =
                (reference.status == "OPTIMAL" && result.status == sihps::MilpStatus::OPTIMAL) ||
                (reference.status == "INFEASIBLE" && result.status == sihps::MilpStatus::INFEASIBLE);
            const double error = result.has_incumbent
                                     ? std::fabs(result.objective_value - reference.objective)
                                     : std::numeric_limits<double>::infinity();
            const bool objective_match = reference.status == "INFEASIBLE" ||
                                         (result.has_incumbent && error <= kObjectiveTolerance);
            const bool exact = status_match && objective_match;
            if (objective_match && reference.status == "OPTIMAL") ++incumbent_matches;
            if (result.status == sihps::MilpStatus::OPTIMAL ||
                result.status == sihps::MilpStatus::INFEASIBLE) {
                ++certified_results;
            } else {
                ++limits;
            }
            if (exact) ++exact_matches;
            else ++mismatches;

            const char* verdict = exact
                                      ? "EXACT"
                                      : (reference.status == "BEST"
                                             ? "NOT_CERTIFIED"
                                             : (objective_match ? "INCUMBENT_ONLY" : "MISMATCH"));
            std::cout << std::left << std::setw(18) << name << std::right << std::setw(12)
                      << status_name(result.status) << std::setw(18) << std::setprecision(12)
                      << (result.has_incumbent ? result.objective_value : 0.0) << " "
                      << std::setw(18)
                      << (reference.status == "INFEASIBLE" ? 0.0 : reference.objective) << " "
                      << std::setw(12) << (std::isfinite(error) ? error : 0.0) << " "
                      << std::setw(10)
                      << result.nodes_processed << std::setw(10) << result.lp_relaxations
                      << std::setw(10) << std::fixed << std::setprecision(3) << seconds
                      << std::setw(10) << cpu_seconds << std::setw(9) << cpu_utilization
                      << std::setw(12) << rss_mb << std::setw(12) << gpu_before_mb << " -> "
                      << std::setw(8) << gpu_after_mb << std::setw(8) << result.cover_cuts
                      << std::setw(8) << result.root_gmi_cuts
                      << std::setw(12) << std::setprecision(8) << result.best_bound << " "
                      << std::setw(10) << result.relative_gap << std::setw(10)
                      << result.warm_started_relaxations << std::setw(10)
                      << result.warm_start_verification_fallbacks << std::right
                      << "  " << verdict
                      << '\n';
        } catch (const std::exception& error) {
            ++mismatches;
            std::cout << std::left << std::setw(18) << name << " ERROR " << error.what() << '\n';
        }
    }

    std::cout << "\nExact reference matches: " << exact_matches << '/' << instances.size() << '\n';
    std::cout << "Optimal-objective incumbent matches: " << incumbent_matches << '/'
              << instances.size() << '\n';
    std::cout << "Certified solver results: " << certified_results << '/' << instances.size() << '\n';
    std::cout << "Time/node-limit results: " << limits << '/' << instances.size() << '\n';
    std::cout << "Mismatches or non-exact results: " << mismatches << '/' << instances.size() << '\n';
    return mismatches == 0 ? 0 : 1;
}

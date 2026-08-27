#include "RunMetadata.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SIHPS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace sihps {
namespace bench {

namespace {

// Values baked in by CMake. Defined defensively so this file still compiles
// if it is ever built outside the project's own build system.
#ifndef SIHPS_GIT_COMMIT
#define SIHPS_GIT_COMMIT "unknown"
#endif
#ifndef SIHPS_GIT_DIRTY
#define SIHPS_GIT_DIRTY "unknown"
#endif
#ifndef SIHPS_BUILD_TYPE
#define SIHPS_BUILD_TYPE "unknown"
#endif

std::string compiler_string() {
    std::ostringstream os;
#if defined(__clang__)
    os << "Clang " << __clang_major__ << '.' << __clang_minor__ << '.' << __clang_patchlevel__;
#elif defined(__GNUC__)
    os << "GNU " << __GNUC__ << '.' << __GNUC_MINOR__ << '.' << __GNUC_PATCHLEVEL__;
#elif defined(_MSC_VER)
    os << "MSVC " << _MSC_VER;
#else
    os << "unknown";
#endif
    return os.str();
}

// Reads one whitespace-delimited field from a /proc file. Linux-only by
// design: this project builds and benchmarks under WSL, and a wrong guess
// on another platform is worse than an honest "unknown".
std::string first_line_after(const char* path, const char* key) {
    std::ifstream in(path);
    if (!in) return "unknown";
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind(key, 0) != 0) continue;
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string value = line.substr(colon + 1);
        const std::size_t begin = value.find_first_not_of(" \t");
        if (begin == std::string::npos) return "unknown";
        const std::size_t end = value.find_last_not_of(" \t\r\n");
        return value.substr(begin, end - begin + 1);
    }
    return "unknown";
}

} // namespace

RunMetadata RunMetadata::capture() {
    RunMetadata m;
    m.git_commit = SIHPS_GIT_COMMIT;
    m.git_dirty = SIHPS_GIT_DIRTY;
    m.build_type = SIHPS_BUILD_TYPE;
    m.compiler = compiler_string();

    m.cpu_model = first_line_after("/proc/cpuinfo", "model name");
    m.host_ram = first_line_after("/proc/meminfo", "MemTotal");

#ifdef _OPENMP
    m.thread_count = omp_get_max_threads();
    {
        omp_sched_t kind;
        int chunk = 0;
        omp_get_schedule(&kind, &chunk);
        std::ostringstream os;
        os << "kind=" << static_cast<int>(kind) << " chunk=" << chunk;
        m.openmp_schedule = os.str();
    }
#else
    m.thread_count = 1;
    m.openmp_schedule = "disabled";
#endif

#ifdef SIHPS_ENABLE_CUDA
    {
        std::ostringstream os;
        os << (CUDART_VERSION / 1000) << '.' << ((CUDART_VERSION % 1000) / 10);
        m.cuda_version = os.str();
    }
    int driver = 0;
    if (cudaDriverGetVersion(&driver) == cudaSuccess) {
        std::ostringstream os;
        os << (driver / 1000) << '.' << ((driver % 1000) / 10);
        m.gpu_driver = os.str();
    } else {
        m.gpu_driver = "unknown";
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        m.gpu_name = prop.name;
        std::ostringstream os;
        os << prop.major << '.' << prop.minor;
        m.gpu_compute_capability = os.str();
    } else {
        m.gpu_name = "unknown";
        m.gpu_compute_capability = "unknown";
    }
#else
    m.cuda_version = "disabled";
    m.gpu_driver = "disabled";
    m.gpu_name = "disabled";
    m.gpu_compute_capability = "disabled";
#endif
    return m;
}

std::string hash_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "unreadable";

    std::uint64_t h = 1469598103934665603ull; // FNV-1a 64 offset basis
    char buffer[65536];
    while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0) {
        const std::streamsize got = in.gcount();
        for (std::streamsize i = 0; i < got; ++i) {
            h ^= static_cast<unsigned char>(buffer[i]);
            h *= 1099511628211ull; // FNV prime
        }
        if (!in) break;
    }

    char out[17];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(h));
    return std::string(out);
}

namespace {

// Minimal JSON string escaping. Instance paths are the only field that can
// realistically contain a backslash, and on Windows hosts they always do.
std::string esc(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// %.17g round-trips an IEEE double exactly. Non-finite values are not legal
// JSON numbers, so they are emitted as strings rather than as bare NaN --
// which would produce a file no standard parser can read.
std::string num(double v) {
    if (!std::isfinite(v)) return std::string("\"") + (std::isnan(v) ? "nan" : (v > 0 ? "inf" : "-inf")) + "\"";
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

} // namespace

JsonlWriter::JsonlWriter(const std::string& path, const RunMetadata& meta,
                          const RunConfig& config) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    file_ = f;
    if (f == nullptr) return;

    std::fprintf(f,
                 "{\"record\":\"header\","
                 "\"git_commit\":\"%s\",\"git_dirty\":\"%s\",\"build_type\":\"%s\","
                 "\"compiler\":\"%s\",\"cuda_version\":\"%s\",\"gpu_name\":\"%s\","
                 "\"gpu_compute_capability\":\"%s\",\"gpu_driver\":\"%s\","
                 "\"cpu_model\":\"%s\",\"host_ram\":\"%s\",\"thread_count\":%d,"
                 "\"openmp_schedule\":\"%s\","
                 "\"method\":\"%s\",\"pricing_rule\":\"%s\",\"algorithm\":\"%s\","
                 "\"presolve\":%s,\"ruiz_scaling\":%s,"
                 "\"hybrid_simplex_budget_seconds\":%s,\"hybrid_first_order_eps\":%s,"
                 "\"row_cap\":%d}\n",
                 esc(meta.git_commit).c_str(), esc(meta.git_dirty).c_str(),
                 esc(meta.build_type).c_str(), esc(meta.compiler).c_str(),
                 esc(meta.cuda_version).c_str(), esc(meta.gpu_name).c_str(),
                 esc(meta.gpu_compute_capability).c_str(), esc(meta.gpu_driver).c_str(),
                 esc(meta.cpu_model).c_str(), esc(meta.host_ram).c_str(), meta.thread_count,
                 esc(meta.openmp_schedule).c_str(), esc(config.method).c_str(),
                 esc(config.pricing_rule).c_str(), esc(config.algorithm).c_str(),
                 config.presolve ? "true" : "false", config.ruiz_scaling ? "true" : "false",
                 num(config.hybrid_simplex_budget_seconds).c_str(),
                 num(config.hybrid_first_order_eps).c_str(), config.row_cap);
    std::fflush(f);
}

JsonlWriter::~JsonlWriter() {
    if (file_ != nullptr) std::fclose(static_cast<std::FILE*>(file_));
}

void JsonlWriter::write(const InstanceRecord& r) {
    if (file_ == nullptr) return;
    std::FILE* f = static_cast<std::FILE*>(file_);
    std::fprintf(f,
                 "{\"record\":\"instance\","
                 "\"instance\":\"%s\",\"hash\":\"%s\",\"rows\":%d,\"cols\":%d,\"nnz\":%d,"
                 "\"status\":\"%s\",\"objective\":%s,\"reference_objective\":%s,"
                 "\"relative_objective_error\":%s,\"passed\":%s,\"reference_source\":\"%s\","
                 "\"wall_seconds\":%s,\"wall_seconds_min\":%s,\"wall_seconds_max\":%s,"
                 "\"repeat_count\":%d,\"repeats_deterministic\":%s,"
                 "\"presolve_seconds\":%s,\"solve_seconds\":%s,"
                 "\"simplex_seconds\":%s,\"first_order_seconds\":%s,"
                 "\"iterations\":%d,\"refactorizations\":%d,"
                 "\"primal_residual\":%s,\"dual_residual\":%s,"
                 "\"used_first_order\":%s,\"first_order_fallback_used\":%s,"
                 "\"pdlp_iterations\":%d,\"pdlp_host_syncs\":%d,"
                 "\"presolve_removed_rows\":%d,\"presolve_removed_cols\":%d,"
                 "\"peak_rss_kb\":%ld,\"gpu_available\":%s,\"gpu_used_mb\":%s}\n",
                 esc(r.instance_path).c_str(), esc(r.instance_hash).c_str(), r.rows, r.cols, r.nnz,
                 esc(r.status).c_str(), num(r.objective).c_str(),
                 num(r.reference_objective).c_str(), num(r.relative_objective_error).c_str(),
                 r.passed ? "true" : "false", esc(r.reference_source).c_str(),
                 num(r.wall_seconds).c_str(), num(r.wall_seconds_min).c_str(),
                 num(r.wall_seconds_max).c_str(), r.repeat_count,
                 r.repeats_deterministic ? "true" : "false", num(r.presolve_seconds).c_str(),
                 num(r.solve_seconds).c_str(), num(r.simplex_seconds).c_str(),
                 num(r.first_order_seconds).c_str(), r.iterations, r.refactorizations,
                 num(r.primal_residual).c_str(), num(r.dual_residual).c_str(),
                 r.used_first_order ? "true" : "false",
                 r.first_order_fallback_used ? "true" : "false", r.pdlp_iterations,
                 r.pdlp_host_syncs, r.presolve_removed_rows, r.presolve_removed_cols,
                 r.peak_rss_kb, r.gpu_available ? "true" : "false", num(r.gpu_used_mb).c_str());
    // Flushed per record so a sweep killed by a time limit still leaves
    // every finished instance on disk.
    std::fflush(f);
}

Summary summarize(const std::vector<InstanceRecord>& records) {
    Summary s;
    s.attempted = static_cast<std::int32_t>(records.size());

    std::vector<double> times;
    double log_sum = 0.0;
    for (const InstanceRecord& r : records) {
        if (!r.passed) continue;
        ++s.solved;
        s.total_seconds += r.wall_seconds;
        s.total_iterations += r.iterations;
        s.worst_relative_objective_error =
            std::max(s.worst_relative_objective_error, r.relative_objective_error);
        times.push_back(r.wall_seconds);
        // Shifted geometric mean: a 0.0 s solve would otherwise take the
        // whole product to zero and report a meaningless average.
        log_sum += std::log(std::max(r.wall_seconds, 1e-3));
    }
    if (times.empty()) return s;

    std::sort(times.begin(), times.end());
    s.geometric_mean_seconds = std::exp(log_sum / static_cast<double>(times.size()));
    s.median_seconds = times[times.size() / 2];
    const std::size_t idx95 =
        static_cast<std::size_t>(0.95 * static_cast<double>(times.size() - 1) + 0.5);
    s.p95_seconds = times[idx95];
    s.max_seconds = times.back();
    return s;
}

} // namespace bench
} // namespace sihps

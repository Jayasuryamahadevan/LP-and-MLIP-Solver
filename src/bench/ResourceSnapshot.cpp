#include "ResourceSnapshot.hpp"

#include "../cuda/CudaDevice.hpp"

#ifdef __linux__
#include <sys/resource.h>
#endif

namespace sihps {
namespace bench {

ResourceSnapshot capture_resources() {
    ResourceSnapshot snapshot;
#ifdef __linux__
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        snapshot.cpu_seconds = static_cast<double>(usage.ru_utime.tv_sec) +
                                static_cast<double>(usage.ru_utime.tv_usec) / 1e6 +
                                static_cast<double>(usage.ru_stime.tv_sec) +
                                static_cast<double>(usage.ru_stime.tv_usec) / 1e6;
        snapshot.peak_rss_kb = usage.ru_maxrss;
    }
#endif
    try {
        if (CudaDevice::device_count() > 0) {
            snapshot.gpu_available = true;
            CudaDevice::select(0);
            CudaDevice::memory_info(snapshot.gpu_free_bytes, snapshot.gpu_total_bytes);
        }
    } catch (...) {
        snapshot.gpu_available = false;
    }
    return snapshot;
}

} // namespace bench
} // namespace sihps

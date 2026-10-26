#include "parallel_utils.hpp"

namespace cis6007 {
namespace parallel_utils {

int get_optimal_thread_count(size_t workload_size, int min_threads) {
    // Get hardware concurrency
    int hardware_threads = std::thread::hardware_concurrency();
    
    // If workload is small, use fewer threads to avoid overhead
    if (workload_size < 1000) {
        return std::max(min_threads, std::min(hardware_threads, 2));
    }
    
    // For medium workloads, use a fraction of available threads
    if (workload_size < 10000) {
        return std::max(min_threads, std::min(hardware_threads, hardware_threads / 2));
    }
    
    // For large workloads, use all available threads
    return std::max(min_threads, hardware_threads);
}

} // namespace parallel_utils
} // namespace cis6007

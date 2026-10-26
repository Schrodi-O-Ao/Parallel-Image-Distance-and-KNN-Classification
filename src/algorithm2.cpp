#include "algorithm2.hpp"
#include "algorithm1.hpp"

namespace cis6007 {
namespace algorithm2 {

// Processes [start_idx, end_idx); end_idx is exclusive. Assumes end_idx <= array1.size().
void parallel_worker_1d(
    size_t start_idx,
    size_t end_idx,
    const PixelArray1D& array1,
    const PixelArray1D& array2,
    std::atomic<double>& partial_sum
) {
    double local_sum = 0.0;
    
    // Process assigned chunk: compute squared differences using Algorithm 1
    for (size_t i = start_idx; i < end_idx; ++i) {
        // Use Algorithm 1: pixel_difference computes (x - y)²
        local_sum += algorithm1::pixel_difference(array1[i], array2[i]);
    }
    
    // Atomically add local sum to global accumulator
    // This ensures thread-safe accumulation without race conditions
    partial_sum.fetch_add(local_sum, std::memory_order_relaxed);
}

double euclidean_distance_1d_parallel(
    const PixelArray1D& array1,
    const PixelArray1D& array2,
    int num_threads
) {
    // Validate input: arrays must have same size
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same size");
    }
    
    if (array1.empty()) {
        return 0.0;  // Empty arrays have zero distance
    }
    
    // Adjust thread count if array is smaller than thread count
    num_threads = std::min(num_threads, static_cast<int>(array1.size()));
    if (num_threads <= 0) {
        num_threads = 1;
    }
    
    const size_t n = array1.size();
    const size_t chunk_size = n / num_threads;
    
    // Atomic accumulator for thread-safe sum of squared differences
    // Using atomic ensures no race conditions when threads update the sum
    std::atomic<double> sum_squared(0.0);
    
    // Vector to hold thread objects
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    // Launch worker threads
    // Each thread processes a distinct chunk of the array
    for (int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        // Last thread handles remaining elements
        size_t end_idx = (t == num_threads - 1) ? n : (t + 1) * chunk_size;
        
        threads.emplace_back(parallel_worker_1d, start_idx, end_idx, 
                           std::cref(array1), std::cref(array2), 
                           std::ref(sum_squared));
    }
    
    // Synchronization: join all threads
    // This ensures main thread waits for all workers to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Compute Euclidean distance: sqrt of sum of squared differences
    // This is the final Ed1 value as specified in Algorithm 2
    double total_sum = sum_squared.load(std::memory_order_relaxed);
    return utils::safe_sqrt(total_sum);
}

double euclidean_distance_1d_sequential(
    const PixelArray1D& array1,
    const PixelArray1D& array2
) {
    // Validate input
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same size");
    }
    
    if (array1.empty()) {
        return 0.0;
    }
    
    // Sequential computation: single thread processes entire array
    double sum_squared = 0.0;
    const size_t n = array1.size();
    
    for (size_t i = 0; i < n; ++i) {
        // Use Algorithm 1 for pixel-wise squared difference
        sum_squared += algorithm1::pixel_difference(array1[i], array2[i]);
    }
    
    // Return Euclidean distance
    return utils::safe_sqrt(sum_squared);
}

double sum_squared_differences_1d_parallel(
    const PixelArray1D& array1,
    const PixelArray1D& array2,
    int num_threads
) {
    // Similar to euclidean_distance_1d_parallel but without sqrt
    // Useful for comparisons where only relative ordering matters
    
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same size");
    }
    
    if (array1.empty()) {
        return 0.0;
    }
    
    num_threads = std::min(num_threads, static_cast<int>(array1.size()));
    if (num_threads <= 0) {
        num_threads = 1;
    }
    
    const size_t n = array1.size();
    const size_t chunk_size = n / num_threads;
    
    std::atomic<double> sum_squared(0.0);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = (t == num_threads - 1) ? n : (t + 1) * chunk_size;
        
        threads.emplace_back(parallel_worker_1d, start_idx, end_idx,
                           std::cref(array1), std::cref(array2),
                           std::ref(sum_squared));
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return sum_squared.load(std::memory_order_relaxed);
}

#ifdef USE_MPI
double euclidean_distance_1d_mpi(
    const PixelArray1D& array1,
    const PixelArray1D& array2
) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same size");
    }
    
    const size_t n = array1.size();
    const size_t local_size = n / size;
    const size_t start_idx = rank * local_size;
    const size_t end_idx = (rank == size - 1) ? n : (rank + 1) * local_size;
    
    // Each process computes its local sum
    double local_sum = 0.0;
    for (size_t i = start_idx; i < end_idx; ++i) {
        local_sum += algorithm1::pixel_difference(array1[i], array2[i]);
    }
    
    // MPI_Allreduce: combine all local sums into global sum
    // This is a collective operation that synchronizes all processes
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Only rank 0 returns the final result
    if (rank == 0) {
        return utils::safe_sqrt(global_sum);
    }
    return 0.0;
}
#endif

#ifdef USE_MPI
double euclidean_distance_1d_hybrid(
    const PixelArray1D& array1,
    const PixelArray1D& array2,
    int threads_per_process
) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same size");
    }
    
    const size_t n = array1.size();
    const size_t local_size = n / size;
    const size_t start_idx = rank * local_size;
    const size_t end_idx = (rank == size - 1) ? n : (rank + 1) * local_size;
    
    // Use threads within each MPI process
    threads_per_process = std::min(threads_per_process, static_cast<int>(end_idx - start_idx));
    if (threads_per_process <= 0) {
        threads_per_process = 1;
    }
    
    const size_t chunk_size = (end_idx - start_idx) / threads_per_process;
    std::atomic<double> local_sum(0.0);
    std::vector<std::thread> threads;
    threads.reserve(threads_per_process);
    
    for (int t = 0; t < threads_per_process; ++t) {
        size_t t_start = start_idx + t * chunk_size;
        size_t t_end = (t == threads_per_process - 1) ? end_idx : start_idx + (t + 1) * chunk_size;
        
        threads.emplace_back(parallel_worker_1d, t_start, t_end,
                           std::cref(array1), std::cref(array2),
                           std::ref(local_sum));
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Combine results across MPI processes
    double final_local_sum = local_sum.load(std::memory_order_relaxed);
    double global_sum;
    MPI_Allreduce(&final_local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        return utils::safe_sqrt(global_sum);
    }
    return 0.0;
}
#endif

} // namespace algorithm2
} // namespace cis6007

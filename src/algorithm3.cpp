#include "algorithm3.hpp"
#include "algorithm2.hpp"
#include "algorithm1.hpp"

namespace cis6007 {
namespace algorithm3 {

// Processes row range [start_row, end_row); end_row is exclusive. Assumes end_row <= array1.size().
void parallel_worker_2d(
    size_t start_row,
    size_t end_row,
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    std::atomic<double>& partial_sum
) {
    double local_sum = 0.0;

    // Process assigned rows - each row processed using Algorithm 2
    // This maintains the chain: Algorithm 3 -> Algorithm 2 -> Algorithm 1
    for (size_t row = start_row; row < end_row; ++row) {
        // Call Algorithm 2 to compute 1D Euclidean distance for this row
        // euclidean_distance_1d_sequential returns sqrt(sum_squared)
        double row_distance = algorithm2::euclidean_distance_1d_sequential(
            array1[row], array2[row]
        );
        // Square to get sum of squared differences for this row
        local_sum += row_distance * row_distance;
    }

    // Atomically add local sum to global accumulator
    partial_sum.fetch_add(local_sum, std::memory_order_relaxed);
}

double euclidean_distance_2d_parallel(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int num_threads
) {
    // Validate input: arrays must have same dimensions
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same number of rows");
    }
    
    if (array1.empty()) {
        return 0.0;
    }
    
    // Check that all rows have the same width
    const size_t width = array1[0].size();
    for (const auto& row : array1) {
        if (row.size() != width) {
            throw std::invalid_argument("All rows in array1 must have the same width");
        }
    }
    for (const auto& row : array2) {
        if (row.size() != width) {
            throw std::invalid_argument("All rows in array2 must have the same width");
        }
    }
    
    // Adjust thread count if array is smaller than thread count
    num_threads = std::min(num_threads, static_cast<int>(array1.size()));
    if (num_threads <= 0) {
        num_threads = 1;
    }
    
    const size_t height = array1.size();
    const size_t rows_per_thread = height / num_threads;
    
    // Atomic accumulator for thread-safe sum of squared differences
    std::atomic<double> sum_squared(0.0);
    
    // Vector to hold thread objects
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    // Launch worker threads
    // Each thread processes a distinct set of rows
    for (int t = 0; t < num_threads; ++t) {
        size_t start_row = t * rows_per_thread;
        // Last thread handles remaining rows
        size_t end_row = (t == num_threads - 1) ? height : (t + 1) * rows_per_thread;
        
        threads.emplace_back(parallel_worker_2d, start_row, end_row,
                           std::cref(array1), std::cref(array2),
                           std::ref(sum_squared));
    }
    
    // Synchronization: join all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Compute 2D Euclidean distance: sqrt of sum of all squared differences
    double total_sum = sum_squared.load(std::memory_order_relaxed);
    return utils::safe_sqrt(total_sum);
}

double euclidean_distance_2d_sequential(
    const PixelArray2D& array1,
    const PixelArray2D& array2
) {
    // Validate input
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same number of rows");
    }
    
    if (array1.empty()) {
        return 0.0;
    }
    
    const size_t width = array1[0].size();
    for (const auto& row : array1) {
        if (row.size() != width) {
            throw std::invalid_argument("All rows in array1 must have the same width");
        }
    }
    for (const auto& row : array2) {
        if (row.size() != width) {
            throw std::invalid_argument("All rows in array2 must have the same width");
        }
    }
    
    // Sequential computation: single thread processes entire 2D array
    double sum_squared = 0.0;

    // Compute sum of squared differences for corresponding elements
    for (size_t row = 0; row < array1.size(); ++row) {
        for (size_t col = 0; col < array1[row].size(); ++col) {
            sum_squared += algorithm1::pixel_difference(array1[row][col], array2[row][col]);
        }
    }
    
    return utils::safe_sqrt(sum_squared);
}

double euclidean_distance_2d_row_parallel(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int num_threads
) {
    // Validate input
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same number of rows");
    }
    
    if (array1.empty()) {
        return 0.0;
    }
    
    const size_t width = array1[0].size();
    for (const auto& row : array1) {
        if (row.size() != width) {
            throw std::invalid_argument("All rows in array1 must have the same width");
        }
    }
    for (const auto& row : array2) {
        if (row.size() != width) {
            throw std::invalid_argument("All rows in array2 must have the same width");
        }
    }
    
    // Row-level parallelism: each thread processes one or more rows
    num_threads = std::min(num_threads, static_cast<int>(array1.size()));
    if (num_threads <= 0) {
        num_threads = 1;
    }
    
    const size_t height = array1.size();
    const size_t rows_per_thread = height / num_threads;
    
    std::atomic<double> sum_squared(0.0);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        size_t start_row = t * rows_per_thread;
        size_t end_row = (t == num_threads - 1) ? height : (t + 1) * rows_per_thread;
        
        threads.emplace_back(parallel_worker_2d, start_row, end_row,
                           std::cref(array1), std::cref(array2),
                           std::ref(sum_squared));
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    double total_sum = sum_squared.load(std::memory_order_relaxed);
    return utils::safe_sqrt(total_sum);
}

double euclidean_distance_2d_nested_parallel(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int row_threads,
    int pixel_threads_per_row
) {
    // Validate input
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same number of rows");
    }
    
    if (array1.empty()) {
        return 0.0;
    }
    
    const size_t width = array1[0].size();
    for (const auto& row : array1) {
        if (row.size() != width) {
            throw std::invalid_argument("All rows in array1 must have the same width");
        }
    }
    for (const auto& row : array2) {
        if (row.size() != width) {
            throw std::invalid_argument("All rows in array2 must have the same width");
        }
    }
    
    // Nested parallelism: use Algorithm 2 (which is parallel) for each row
    row_threads = std::min(row_threads, static_cast<int>(array1.size()));
    if (row_threads <= 0) {
        row_threads = 1;
    }
    
    const size_t height = array1.size();
    const size_t rows_per_thread = height / row_threads;
    
    std::atomic<double> sum_squared(0.0);
    std::vector<std::thread> threads;
    threads.reserve(row_threads);
    
    // Lambda for processing rows with nested parallelism
    auto row_processor = [&](size_t start_row, size_t end_row) {
        double local_sum = 0.0;
        for (size_t row = start_row; row < end_row; ++row) {
            // Call Algorithm 2 with pixel_threads_per_row for each row
            // This creates nested parallelism
            double row_distance = algorithm2::euclidean_distance_1d_parallel(
                array1[row], array2[row], pixel_threads_per_row
            );
            // Square the distance to get sum of squared differences for this row
            local_sum += row_distance * row_distance;
        }
        sum_squared.fetch_add(local_sum, std::memory_order_relaxed);
    };
    
    for (int t = 0; t < row_threads; ++t) {
        size_t start_row = t * rows_per_thread;
        size_t end_row = (t == row_threads - 1) ? height : (t + 1) * rows_per_thread;
        
        threads.emplace_back(row_processor, start_row, end_row);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    double total_sum = sum_squared.load(std::memory_order_relaxed);
    return utils::safe_sqrt(total_sum);
}

#ifdef USE_MPI
double euclidean_distance_2d_mpi(
    const PixelArray2D& array1,
    const PixelArray2D& array2
) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same number of rows");
    }

    const size_t height = array1.size();
    const size_t local_rows = height / size;
    const size_t start_row = rank * local_rows;
    const size_t end_row = (rank == size - 1) ? height : (rank + 1) * local_rows;

    // Each process computes its local sum using Algorithm 2 per row
    // Maintains chain: Algorithm 3 -> Algorithm 2 -> Algorithm 1
    double local_sum = 0.0;
    for (size_t row = start_row; row < end_row; ++row) {
        // Call Algorithm 2 to compute 1D Euclidean distance for this row
        double row_distance = algorithm2::euclidean_distance_1d_sequential(
            array1[row], array2[row]
        );
        // Square to get sum of squared differences for this row
        local_sum += row_distance * row_distance;
    }

    // Combine results across MPI processes
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        return utils::safe_sqrt(global_sum);
    }
    return 0.0;
}
#endif

#ifdef USE_MPI
double euclidean_distance_2d_hybrid(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int threads_per_process
) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (array1.size() != array2.size()) {
        throw std::invalid_argument("Arrays must have the same number of rows");
    }
    
    const size_t height = array1.size();
    const size_t local_rows = height / size;
    const size_t start_row = rank * local_rows;
    const size_t end_row = (rank == size - 1) ? height : (rank + 1) * local_rows;
    
    // Use threads within each MPI process
    threads_per_process = std::min(threads_per_process, static_cast<int>(end_row - start_row));
    if (threads_per_process <= 0) {
        threads_per_process = 1;
    }
    
    const size_t rows_per_thread = (end_row - start_row) / threads_per_process;
    std::atomic<double> local_sum(0.0);
    std::vector<std::thread> threads;
    threads.reserve(threads_per_process);
    
    for (int t = 0; t < threads_per_process; ++t) {
        size_t t_start = start_row + t * rows_per_thread;
        size_t t_end = (t == threads_per_process - 1) ? end_row : start_row + (t + 1) * rows_per_thread;
        
        threads.emplace_back(parallel_worker_2d, t_start, t_end,
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

} // namespace algorithm3
} // namespace cis6007

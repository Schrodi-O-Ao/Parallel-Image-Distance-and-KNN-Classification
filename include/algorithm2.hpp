#ifndef ALGORITHM2_HPP
#define ALGORITHM2_HPP

/**
 * @file algorithm2.hpp
 * @brief Algorithm 2: Compare two 1D pixel arrays using parallel computation.
 * @author ST20290339
 */

#include "common.hpp"

namespace cis6007 {
namespace algorithm2 {

/**
 * @brief Compare two 1D pixel arrays using parallel computation (Algorithm 2).
 *
 * Computes the Euclidean distance Ed1 = sqrt(sum((x[i] - y[i])^2)).
 * This algorithm extends Algorithm 1 to work on 1D arrays by:
 * 1. Computing squared differences for each corresponding pixel pair
 * 2. Summing all squared differences across worker threads
 * 3. Taking the square root to get Euclidean distance
 *
 * @param array1 First 1D pixel array.
 * @param array2 Second 1D pixel array (must match size of array1).
 * @param num_threads Number of worker threads to use (default: hardware concurrency).
 * @return Euclidean distance Ed1 between the two arrays.
 *
 * @throws std::invalid_argument if arrays have different sizes.
 * @note Complexity and scaling:
 *       - Time complexity: O(n) where n is array length.
 *       - Space complexity: O(1) additional space.
 *       - Parallel speedup approaches num_threads for large arrays.
 */
double euclidean_distance_1d_parallel(
    const PixelArray1D& array1,
    const PixelArray1D& array2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Sequential version of 1D Euclidean distance computation.
 *
 * Used for benchmarking, testing, and validation of the parallel version.
 *
 * @param array1 First 1D pixel array.
 * @param array2 Second 1D pixel array.
 * @return Euclidean distance Ed1.
 *
 * @throws std::invalid_argument if arrays have different sizes.
 */
double euclidean_distance_1d_sequential(
    const PixelArray1D& array1,
    const PixelArray1D& array2
);

/**
 * @brief Compute sum of squared differences without taking the square root.
 *
 * Useful for nearest-neighbor comparisons where relative ordering suffices.
 *
 * @param array1 First 1D pixel array.
 * @param array2 Second 1D pixel array.
 * @param num_threads Number of threads for parallel computation.
 * @return Sum of squared differences.
 * @throws std::invalid_argument if arrays have different sizes.
 */
double sum_squared_differences_1d_parallel(
    const PixelArray1D& array1,
    const PixelArray1D& array2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief MPI-enabled version of 1D Euclidean distance.
 *
 * Distributes computation across MPI processes.
 *
 * @param array1 First 1D pixel array.
 * @param array2 Second 1D pixel array.
 * @return Euclidean distance Ed1.
 *
 * @note Only available when USE_MPI is defined; requires MPI initialization before calling.
 */
#ifdef USE_MPI
double euclidean_distance_1d_mpi(
    const PixelArray1D& array1,
    const PixelArray1D& array2
);
#endif

/**
 * @brief Hybrid MPI + OpenMP/Thread version for distributed and multi-core parallelism.
 *
 * Uses MPI for inter-node communication and threads for intra-node parallelism.
 *
 * @param array1 First 1D pixel array.
 * @param array2 Second 1D pixel array.
 * @param threads_per_process Number of threads per MPI process.
 * @return Euclidean distance Ed1.
 *
 * @note Only available when USE_MPI is defined.
 */
#ifdef USE_MPI
double euclidean_distance_1d_hybrid(
    const PixelArray1D& array1,
    const PixelArray1D& array2,
    int threads_per_process = std::thread::hardware_concurrency()
);
#endif

} // namespace algorithm2
} // namespace cis6007

#endif // ALGORITHM2_HPP

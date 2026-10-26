#ifndef ALGORITHM3_HPP
#define ALGORITHM3_HPP

/**
 * @file algorithm3.hpp
 * @brief Algorithm 3: Compare two 2D pixel arrays (images) using parallel computation.
 * @author ST20290339
 */

#include "common.hpp"

namespace cis6007 {
namespace algorithm3 {

/**
 * @brief Compare two 2D pixel arrays using parallel computation (Algorithm 3).
 *
 * Computes the 2D Euclidean distance Ed2 by calling Algorithm 2 row-by-row in parallel.
 * This algorithm extends Algorithm 2 to work on 2D arrays (images) by:
 * 1. Treating each row as a 1D array
 * 2. Computing 1D Euclidean distance for each row using Algorithm 2
 * 3. Combining row distances using the Pythagorean theorem in 2D
 *
 * @param array1 First 2D pixel array (image).
 * @param array2 Second 2D pixel array (must match dimensions of array1).
 * @param num_threads Number of worker threads for row-level parallelism (default: hardware concurrency).
 * @return 2D Euclidean distance Ed2 between the two arrays.
 *
 * @throws std::invalid_argument if arrays have different dimensions or ragged rows.
 * @note Complexity:
 *       - Time complexity: O(n*m) where n is height and m is width.
 *       - Space complexity: O(1) additional space.
 */
double euclidean_distance_2d_parallel(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Sequential version of 2D Euclidean distance computation.
 *
 * Used for validation, testing, and performance baseline comparison.
 *
 * @param array1 First 2D pixel array.
 * @param array2 Second 2D pixel array.
 * @return 2D Euclidean distance Ed2.
 *
 * @throws std::invalid_argument if arrays have different dimensions.
 */
double euclidean_distance_2d_sequential(
    const PixelArray2D& array1,
    const PixelArray2D& array2
);

/**
 * @brief Compute 2D Euclidean distance using row-wise parallelism only.
 *
 * Each row is processed sequentially, but rows are distributed across threads.
 *
 * @param array1 First 2D pixel array.
 * @param array2 Second 2D pixel array.
 * @param num_threads Number of worker threads.
 * @return 2D Euclidean distance Ed2.
 */
double euclidean_distance_2d_row_parallel(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Compute 2D Euclidean distance using nested parallelism.
 *
 * Utilizes both row-level and pixel-level parallelism for maximum multi-core speedup.
 *
 * @param array1 First 2D pixel array.
 * @param array2 Second 2D pixel array.
 * @param row_threads Number of threads for row-level parallelism.
 * @param pixel_threads_per_row Number of threads per row for pixel-level parallelism.
 * @return 2D Euclidean distance Ed2.
 */
double euclidean_distance_2d_nested_parallel(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int row_threads = std::thread::hardware_concurrency(),
    int pixel_threads_per_row = 2
);

/**
 * @brief MPI-enabled version of 2D Euclidean distance.
 *
 * Distributes rows across MPI processes.
 *
 * @param array1 First 2D pixel array.
 * @param array2 Second 2D pixel array.
 * @return 2D Euclidean distance Ed2.
 *
 * @note Only available when USE_MPI is defined.
 */
#ifdef USE_MPI
double euclidean_distance_2d_mpi(
    const PixelArray2D& array1,
    const PixelArray2D& array2
);
#endif

/**
 * @brief Hybrid MPI + Thread version for 2D arrays.
 *
 * Uses MPI for row distribution and threads for intra-row parallelism.
 *
 * @param array1 First 2D pixel array.
 * @param array2 Second 2D pixel array.
 * @param threads_per_process Number of threads per MPI process.
 * @return 2D Euclidean distance Ed2.
 *
 * @note Only available when USE_MPI is defined.
 */
#ifdef USE_MPI
double euclidean_distance_2d_hybrid(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int threads_per_process = std::thread::hardware_concurrency()
);
#endif

} // namespace algorithm3
} // namespace cis6007

#endif // ALGORITHM3_HPP

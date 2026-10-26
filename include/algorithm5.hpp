#ifndef ALGORITHM5_HPP
#define ALGORITHM5_HPP

/**
 * @file algorithm5.hpp
 * @brief Percentage-distance (RMS of channel difference) algorithms.
 * @author ST20290339
 */

#include "common.hpp"

namespace cis6007 {
namespace algorithm5 {

/**
 * @brief Compute the RMS percentage distance between two 2D pixel arrays in parallel.
 *
 * Delegates to Algorithm 3, then normalises the Euclidean distance by
 * sqrt(width * height * channels) and 255, yielding a value in [0, 100].
 *
 * @param array1 First pixel array.
 * @param array2 Second pixel array (must match dimensions and row widths).
 * @param num_threads Worker threads to use (default: hardware concurrency).
 * @return Root-mean-square difference across all stored channel scalars,
 *         expressed as a percentage of the 8-bit range (0 to 100).
 * @throws std::invalid_argument if dimensions or row widths differ.
 */
double percentage_distance_parallel(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Compute the RMS percentage distance sequentially without worker threads.
 *
 * @param array1 First pixel array.
 * @param array2 Second pixel array.
 * @return Root-mean-square difference across all stored channel scalars (0 to 100).
 * @throws std::invalid_argument if dimensions or row widths differ.
 */
double percentage_distance_sequential(
    const PixelArray2D& array1,
    const PixelArray2D& array2
);

/**
 * @brief Compare equal-sized Image structures using RMS percentage distance.
 *
 * @param image1 First image.
 * @param image2 Second image.
 * @param num_threads Worker threads for parallel execution.
 * @return Root-mean-square difference across all stored channel scalars (0 to 100).
 * @throws std::invalid_argument if dimensions or channel counts differ.
 */
double percentage_distance_images(
    const Image& image1,
    const Image& image2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Return distance and a heuristic score in DistanceResult.
 *
 * Note that confidence is set to 100 minus distance, clamped to [0, 100].
 * It is a heuristic score, not a calibrated probability.
 *
 * @param array1 First pixel array.
 * @param array2 Second pixel array.
 * @param num_threads Worker threads for parallel execution.
 * @return DistanceResult containing percentage distance, confidence, and metadata.
 */
DistanceResult percentage_distance_detailed(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Load two images from disk, resize to common dimensions, and compute RMS percentage distance.
 *
 * @param image_path1 Path to first image.
 * @param image_path2 Path to second image.
 * @param num_threads Worker threads for parallel execution.
 * @return DistanceResult containing percentage distance and metadata.
 * @throws std::runtime_error if either image fails to load.
 */
DistanceResult load_and_compare_percentage(
    const std::string& image_path1,
    const std::string& image_path2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Return 100 minus the percentage distance, clamped to [0, 100].
 *
 * This represents a heuristic similarity score, not a calibrated probability.
 *
 * @param percentage_distance RMS percentage distance in [0, 100].
 * @return Heuristic similarity score in [0, 100].
 */
double percentage_to_similarity(double percentage_distance);

/**
 * @brief Return a qualitative textual label based on fixed distance thresholds.
 *
 * @param percentage_distance RMS percentage distance.
 * @return Qualitative classification string (e.g. Identical, Very Similar).
 */
std::string interpret_percentage_distance(double percentage_distance);

/**
 * @brief Compare one reference array against a batch of candidate arrays in parallel.
 *
 * @param reference_array Reference 2D pixel array.
 * @param comparison_arrays Vector of candidate 2D pixel arrays.
 * @param num_threads Worker threads for outer batch distribution.
 * @return Vector of percentage distance scores for each comparison.
 */
std::vector<double> batch_percentage_distance(
    const PixelArray2D& reference_array,
    const std::vector<PixelArray2D>& comparison_arrays,
    int num_threads = std::thread::hardware_concurrency()
);

} // namespace algorithm5
} // namespace cis6007

#endif // ALGORITHM5_HPP

#ifndef ALGORITHM1_HPP
#define ALGORITHM1_HPP

/**
 * @file algorithm1.hpp
 * @brief Pixel comparison algorithms (squared and absolute difference).
 * @author ST20290339
 */

#include "common.hpp"

namespace cis6007 {
namespace algorithm1 {

/**
 * @brief Algorithm 1: Compare two individual pixels.
 *
 * Computes the squared difference between two pixel values: (x - y)^2.
 * This is the fundamental building block for all higher-level algorithms,
 * providing a measure of dissimilarity between two pixel intensities.
 *
 * @param x First pixel value (typically 0-255 for 8-bit images).
 * @param y Second pixel value (typically 0-255 for 8-bit images).
 * @return Squared difference (x - y)^2, always non-negative.
 *
 * @note Time complexity is O(1) and requires no thread synchronization.
 *       Squared difference maintains consistency with Euclidean distance.
 */
inline double pixel_difference(Pixel x, Pixel y) {
    // Cast to double to avoid integer overflow for large differences
    // Compute (x - y)^2 as specified in the algorithm requirements
    double diff = static_cast<double>(x) - static_cast<double>(y);
    return diff * diff;
}

/**
 * @brief Alternative pixel comparison using integer arithmetic for performance.
 *
 * Computes (x - y)^2 using integer operations.
 *
 * @param x First pixel value.
 * @param y Second pixel value.
 * @return Squared difference as integer.
 *
 * @note The 46340 bound applies only if Pixel is widened beyond uint8_t;
 *       with the current uint8_t typedef this cannot overflow.
 */
inline int pixel_difference_int(Pixel x, Pixel y) {
    int diff = static_cast<int>(x) - static_cast<int>(y);
    return diff * diff;
}

/**
 * @brief Compute absolute difference between two pixels: |x - y|.
 *
 * @param x First pixel value.
 * @param y Second pixel value.
 * @return Absolute difference |x - y|.
 */
inline double pixel_absolute_difference(Pixel x, Pixel y) {
    return std::abs(static_cast<double>(x) - static_cast<double>(y));
}

/**
 * @brief Compute normalized pixel difference in range [0, 1].
 *
 * Normalizes the squared difference by the maximum possible value (255^2).
 *
 * @param x First pixel value.
 * @param y Second pixel value.
 * @return Normalized squared difference in [0, 1].
 */
inline double pixel_difference_normalized(Pixel x, Pixel y) {
    constexpr double MAX_DIFF_SQUARED = 255.0 * 255.0;
    return pixel_difference(x, y) / MAX_DIFF_SQUARED;
}

} // namespace algorithm1
} // namespace cis6007

#endif // ALGORITHM1_HPP

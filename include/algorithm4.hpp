#ifndef ALGORITHM4_HPP
#define ALGORITHM4_HPP

/**
 * @file algorithm4.hpp
 * @brief Algorithm 4: Real-world image loading via OpenCV and comparison via Algorithm 3.
 * @author ST20290339
 */

#include "common.hpp"

namespace cis6007 {
namespace algorithm4 {

/**
 * @brief Algorithm 4: Load two real images into 2D arrays using OpenCV and compare via Algorithm 3.
 *
 * This algorithm bridges theoretical algorithms and real-world image processing by:
 * 1. Loading images from disk using OpenCV
 * 2. Converting OpenCV Mat format to internal PixelArray2D format
 * 3. Handling color spaces (RGB, grayscale)
 * 4. Resizing images to match dimensions if needed
 * 5. Computing distance using Algorithm 3
 *
 * @param image_path1 Path to first image file.
 * @param image_path2 Path to second image file.
 * @param num_threads Number of threads for parallel computation.
 * @return DistanceResult containing Euclidean distance and metadata.
 *
 * @throws std::runtime_error If either image file cannot be read or decoded from disk.
 * @throws std::invalid_argument If image dimensions are incompatible and cannot be matched.
 * @note Supports standard image formats: PNG, JPG, BMP, TIFF.
 */
DistanceResult load_and_compare_images(
    const std::string& image_path1,
    const std::string& image_path2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Load a single image from file into the Image structure.
 *
 * @param image_path Path to image file.
 * @param target_width Optional target width (0 to keep original).
 * @param target_height Optional target height (0 to keep original).
 * @return Image structure containing pixel data.
 *
 * @throws std::runtime_error If image file cannot be found or decoded.
 */
Image load_image(
    const std::string& image_path,
    int target_width = 0,
    int target_height = 0
);

/**
 * @brief Convert OpenCV Mat to internal PixelArray2D format.
 *
 * Handles different color spaces and channel configurations.
 *
 * @param mat OpenCV Mat containing image data.
 * @return PixelArray2D in internal row-major interleaved channel format.
 */
PixelArray2D mat_to_pixel_array(const cv::Mat& mat);

/**
 * @brief Convert internal PixelArray2D back to OpenCV Mat format.
 *
 * @param array Internal pixel array format.
 * @param channels Number of color channels (1 for grayscale, 3 for RGB).
 * @return OpenCV Mat representation.
 */
cv::Mat pixel_array_to_mat(const PixelArray2D& array, int channels = 3);

/**
 * @brief Resize image to specified dimensions.
 *
 * @param image Image to resize.
 * @param new_width Target width.
 * @param new_height Target height.
 * @return Resized image.
 */
Image resize_image(const Image& image, int new_width, int new_height);

/**
 * @brief Convert image to grayscale.
 *
 * @param image Input image (RGB or grayscale).
 * @return Grayscale version of the image.
 */
Image convert_to_grayscale(const Image& image);

/**
 * @brief Compare two Image structures using Algorithm 3.
 *
 * @param image1 First image.
 * @param image2 Second image.
 * @param num_threads Number of worker threads for parallel computation.
 * @return Euclidean distance between images.
 *
 * @throws std::invalid_argument If images have incompatible dimensions or channels.
 */
double compare_images(
    const Image& image1,
    const Image& image2,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Batch image comparison: compare one image against multiple candidates.
 *
 * @param reference_image Image to compare against.
 * @param image_paths Vector of file paths to candidate images.
 * @param num_threads Number of worker threads.
 * @return Vector of DistanceResult for each comparison.
 */
std::vector<DistanceResult> batch_compare_images(
    const Image& reference_image,
    const std::vector<std::string>& image_paths,
    int num_threads = std::thread::hardware_concurrency()
);

/**
 * @brief Save image structure to disk.
 *
 * @param image Image to save.
 * @param output_path File path where image should be saved.
 * @return true if successful, false otherwise.
 */
bool save_image(const Image& image, const std::string& output_path);

/**
 * @brief Get formatted string containing image metadata (dimensions, channels, size).
 *
 * @param image_path Path to image file.
 * @return String containing image information.
 */
std::string get_image_info(const std::string& image_path);

} // namespace algorithm4
} // namespace cis6007

#endif // ALGORITHM4_HPP

#ifndef COMMON_HPP
#define COMMON_HPP

/**
 * @file common.hpp
 * @brief Common data structures, type definitions, and utilities for CIS6007.
 * @author ST20290339
 */

// Standard library includes
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>

// Threading includes
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>

// MPI includes (optional - can be disabled if MPI not available)
#ifdef USE_MPI
#include <mpi.h>
#endif

// OpenCV includes for image processing
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Namespace for the project
namespace cis6007 {

// Type definitions for pixel values
using Pixel = uint8_t;
using PixelArray1D = std::vector<Pixel>;
using PixelArray2D = std::vector<std::vector<Pixel>>;

/**
 * @brief RGBA/RGB/Grayscale image stored row-major with interleaved channels.
 *
 * Pixel at (row, col, channel) is data[row][col * channels + channel].
 */
struct Image {
    PixelArray2D data; ///< 2D pixel array: data[row][col * channels + channel]
    int width;         ///< Image width in pixels
    int height;        ///< Image height in pixels
    int channels;      ///< Interleaved color channels (1 for grayscale, 3 for RGB)
    
    Image() : width(0), height(0), channels(3) {}
    Image(int w, int h, int c = 3) : width(w), height(h), channels(c) {
        data.resize(height, std::vector<Pixel>(width * channels));
    }
};

/**
 * @brief Result of a distance / classification computation.
 *
 * @note Semantics of fields:
 *       - distance: Euclidean distance or RMS percentage difference.
 *         For KNNClassifier::classify, set to the mean percentage distance
 *         over the k nearest training samples.
 *       - confidence: Heuristic score in [0, 100], not a calibrated probability:
 *         * Set to (100 - distance) by algorithm5::percentage_distance_detailed.
 *         * Set to assignment formula (1 - min / (sum * k)) * 100 by KNNClassifier::classify.
 *         * Set to 0.0 for error paths.
 *       - label: Integer class index (0-9 for CIFAR-10, -1 if unclassified).
 *       - label_name: Human-readable class name string.
 */
struct DistanceResult {
    double distance;         ///< Euclidean or mean percentage distance
    double confidence;       ///< Heuristic confidence score in [0, 100]
    int label;               ///< Predicted class label (-1 if unclassified)
    std::string label_name;  ///< Descriptive class label name
    
    DistanceResult() : distance(0.0), confidence(0.0), label(-1) {}
};

/**
 * @brief Fixed 32x32x3 CIFAR-10 image with ground-truth class label.
 *
 * Stored row-major with 3 interleaved channels per pixel (32 rows, 96 elements per row).
 */
struct CIFARImage {
    PixelArray2D data;  ///< 32x32x3 RGB image data
    int label;          ///< Ground-truth class label (0-9)
    
    CIFARImage() : label(0) {
        data.resize(32, std::vector<Pixel>(32 * 3));
    }
};

/**
 * @brief Execution configuration for parallel algorithms and thread scheduling.
 */
struct ParallelConfig {
    int num_threads; ///< Worker thread count (default: hardware concurrency)
    bool use_mpi;    ///< Flag indicating whether MPI distribution is active
    int chunk_size;  ///< Batch workload chunk size per worker
    
    ParallelConfig() : num_threads(std::thread::hardware_concurrency()), 
                      use_mpi(false), 
                      chunk_size(1000) {}
};

// Constants
constexpr int CIFAR_IMAGE_SIZE = 32;
constexpr int CIFAR_CHANNELS = 3;
constexpr int CIFAR_CLASSES = 10;
constexpr int CIFAR_TRAIN_SIZE = 50000;
constexpr int CIFAR_TEST_SIZE = 10000;

// CIFAR-10 class names
inline const std::vector<std::string> CIFAR_CLASS_NAMES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Utility functions
namespace utils {
    // Get current timestamp in milliseconds
    inline double get_timestamp_ms() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
    }
    
    // Convert seconds to milliseconds
    inline double sec_to_ms(double seconds) {
        return seconds * 1000.0;
    }
    
    // Safe square root with error checking
    inline double safe_sqrt(double value) {
        if (value < 0.0) {
            throw std::invalid_argument("Cannot compute square root of negative value");
        }
        return std::sqrt(value);
    }
    
    // Clamp value between min and max
    template<typename T>
    inline T clamp(T value, T min_val, T max_val) {
        return std::max(min_val, std::min(value, max_val));
    }
}

} // namespace cis6007

#endif // COMMON_HPP

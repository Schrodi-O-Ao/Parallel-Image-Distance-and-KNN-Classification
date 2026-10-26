#include "algorithm5.hpp"
#include "algorithm3.hpp"
#include "algorithm4.hpp"
#include <sstream>

namespace cis6007 {
namespace algorithm5 {

namespace {
constexpr double PIXEL_MAX_VALUE = 255.0;

// Converts a Euclidean distance into an RMS percentage of the 8-bit range.
// value_count must be the total number of stored scalars (w*h*channels).
double to_percentage(double euclidean_distance, size_t value_count) {
    if (value_count == 0) return 0.0;
    const double root_mean_square = euclidean_distance / std::sqrt(static_cast<double>(value_count));
    return std::clamp(root_mean_square / PIXEL_MAX_VALUE * 100.0, 0.0, 100.0);
}
}

double percentage_distance_parallel(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int num_threads
) {
    // All overloads count the same stored channel values in their RMS calculation.
    double euclidean_distance = algorithm3::euclidean_distance_2d_parallel(array1, array2, num_threads);
    const size_t value_count = array1.empty() ? 0 : array1.size() * array1.front().size();
    return to_percentage(euclidean_distance, value_count);
}

double percentage_distance_sequential(
    const PixelArray2D& array1,
    const PixelArray2D& array2
) {
    double euclidean_distance = algorithm3::euclidean_distance_2d_sequential(array1, array2);
    const size_t value_count = array1.empty() ? 0 : array1.size() * array1.front().size();
    return to_percentage(euclidean_distance, value_count);
}

double percentage_distance_images(
    const Image& image1,
    const Image& image2,
    int num_threads
) {
    // Require matching image geometry and channel count for element-wise comparison.
    if (image1.width != image2.width || image1.height != image2.height) {
        throw std::invalid_argument("Images must have the same dimensions");
    }

    if (image1.channels != image2.channels) {
        throw std::invalid_argument("Images must have the same number of channels");
    }

    return percentage_distance_parallel(image1.data, image2.data, num_threads);
}

DistanceResult percentage_distance_detailed(
    const PixelArray2D& array1,
    const PixelArray2D& array2,
    int num_threads
) {
    DistanceResult result;
    double euclidean_distance = algorithm3::euclidean_distance_2d_parallel(array1, array2, num_threads);
    const size_t value_count = array1.empty() ? 0 : array1.size() * array1.front().size();
    result.distance = to_percentage(euclidean_distance, value_count);
    result.confidence = 100.0 - result.distance;

    return result;
}

DistanceResult load_and_compare_percentage(
    const std::string& image_path1,
    const std::string& image_path2,
    int num_threads
) {
    // Load images using Algorithm 4
    Image image1 = algorithm4::load_image(image_path1);
    Image image2 = algorithm4::load_image(image_path2);
    
    // Resize both inputs to their shared smaller dimensions before comparison.
    if (image1.width != image2.width || image1.height != image2.height) {
        int target_width = std::min(image1.width, image2.width);
        int target_height = std::min(image1.height, image2.height);
        
        image1 = algorithm4::resize_image(image1, target_width, target_height);
        image2 = algorithm4::resize_image(image2, target_width, target_height);
    }
    
    return percentage_distance_detailed(image1.data, image2.data, num_threads);
}

double percentage_to_similarity(double percentage_distance) {
    // This compatibility helper returns a simple inverse-distance score, not a probability.
    return 100.0 - std::clamp(percentage_distance, 0.0, 100.0);
}

std::string interpret_percentage_distance(double percentage_distance) {
    // Select a fixed text category from the numeric distance thresholds.
    if (percentage_distance < 5.0) {
        return "Very similar - Images are nearly identical";
    } else if (percentage_distance < 15.0) {
        return "Similar - Images have minor differences";
    } else if (percentage_distance < 30.0) {
        return "Moderately similar - Images share some features";
    } else if (percentage_distance < 50.0) {
        return "Different - Images have significant differences";
    } else if (percentage_distance < 100.0) {
        return "Very different - Images share few similarities";
    } else {
        return "Extremely different - Images are unrelated";
    }
}

std::vector<double> batch_percentage_distance(
    const PixelArray2D& reference_array,
    const std::vector<PixelArray2D>& comparison_arrays,
    int num_threads
) {
    // Parallel batch computation
    std::vector<std::thread> threads;
    std::vector<double> thread_results(comparison_arrays.size());
    std::atomic<size_t> current_index(0);
    
    auto worker = [&]() {
        while (true) {
            size_t comparison_index = current_index.fetch_add(1);
            if (comparison_index >= comparison_arrays.size()) break;
            
            try {
                thread_results[comparison_index] = percentage_distance_parallel(
                    reference_array, comparison_arrays[comparison_index], 1
                );
            } catch (const std::exception& e) {
                thread_results[comparison_index] = -1.0;  // Mark this failed comparison for the caller.
            }
        }
    };
    
    // Launch threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker);
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    return thread_results;
}

} // namespace algorithm5
} // namespace cis6007

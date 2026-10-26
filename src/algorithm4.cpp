#include "algorithm4.hpp"
#include "algorithm3.hpp"
#include <sstream>

namespace cis6007 {
namespace algorithm4 {

Image load_image(
    const std::string& image_path,
    int target_width,
    int target_height
) {
    // Load image using OpenCV
    cv::Mat mat = cv::imread(image_path, cv::IMREAD_COLOR);
    
    if (mat.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    
    // Convert BGR (OpenCV default) to RGB
    cv::Mat rgb_mat;
    cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    
    // Resize if target dimensions specified
    if (target_width > 0 && target_height > 0) {
        cv::resize(rgb_mat, rgb_mat, cv::Size(target_width, target_height));
    }
    
    // Convert to our Image structure
    Image image;
    image.width = rgb_mat.cols;
    image.height = rgb_mat.rows;
    image.channels = rgb_mat.channels();
    image.data = mat_to_pixel_array(rgb_mat);
    
    return image;
}

PixelArray2D mat_to_pixel_array(const cv::Mat& mat) {
    PixelArray2D result;
    result.reserve(mat.rows);
    
    // Iterate through each row of the image
    for (int row = 0; row < mat.rows; ++row) {
        std::vector<Pixel> pixel_row;
        pixel_row.reserve(mat.cols * mat.channels());
        
        // Get pointer to row data for efficient access
        const uint8_t* row_ptr = mat.ptr<uint8_t>(row);
        
        // Iterate through each pixel in the row
        for (int col = 0; col < mat.cols; ++col) {
            // Extract each channel (R, G, B for color images)
            for (int channel = 0; channel < mat.channels(); ++channel) {
                pixel_row.push_back(row_ptr[col * mat.channels() + channel]);
            }
        }
        
        result.push_back(pixel_row);
    }
    
    return result;
}

cv::Mat pixel_array_to_mat(const PixelArray2D& array, int channels) {
    if (array.empty()) {
        return cv::Mat();
    }
    
    int height = array.size();
    int width = array[0].size() / channels;
    
    // Create OpenCV Mat
    cv::Mat mat(height, width, CV_8UC(channels));
    
    // Fill Mat with pixel data
    for (int row = 0; row < height; ++row) {
        uint8_t* row_ptr = mat.ptr<uint8_t>(row);
        for (int col = 0; col < width; ++col) {
            for (int channel = 0; channel < channels; ++channel) {
                row_ptr[col * channels + channel] = array[row][col * channels + channel];
            }
        }
    }
    
    return mat;
}

Image resize_image(const Image& image, int new_width, int new_height) {
    // Convert to OpenCV Mat
    cv::Mat mat = pixel_array_to_mat(image.data, image.channels);
    
    // Resize
    cv::Mat resized_mat;
    cv::resize(mat, resized_mat, cv::Size(new_width, new_height));
    
    // Convert back to Image
    Image resized_image;
    resized_image.width = new_width;
    resized_image.height = new_height;
    resized_image.channels = image.channels;
    resized_image.data = mat_to_pixel_array(resized_mat);
    
    return resized_image;
}

Image convert_to_grayscale(const Image& image) {
    if (image.channels == 1) {
        return image;  // Already grayscale
    }
    
    // Convert to OpenCV Mat
    cv::Mat mat = pixel_array_to_mat(image.data, image.channels);
    
    // Convert to grayscale
    cv::Mat gray_mat;
    cv::cvtColor(mat, gray_mat, cv::COLOR_RGB2GRAY);
    
    // Convert back to Image
    Image gray_image;
    gray_image.width = gray_mat.cols;
    gray_image.height = gray_mat.rows;
    gray_image.channels = 1;
    gray_image.data = mat_to_pixel_array(gray_mat);
    
    return gray_image;
}

double compare_images(
    const Image& image1,
    const Image& image2,
    int num_threads
) {
    // Validate dimensions
    if (image1.width != image2.width || image1.height != image2.height) {
        throw std::invalid_argument("Images must have the same dimensions for comparison");
    }
    
    if (image1.channels != image2.channels) {
        throw std::invalid_argument("Images must have the same number of channels");
    }
    
    // Use Algorithm 3 to compute 2D Euclidean distance
    return algorithm3::euclidean_distance_2d_parallel(
        image1.data, image2.data, num_threads
    );
}

DistanceResult load_and_compare_images(
    const std::string& image_path1,
    const std::string& image_path2,
    int num_threads
) {
    DistanceResult result;
    
    try {
        // Load both images
        Image image1 = load_image(image_path1);
        Image image2 = load_image(image_path2);
        
        // Resize to match dimensions if needed
        if (image1.width != image2.width || image1.height != image2.height) {
            int target_width = std::min(image1.width, image2.width);
            int target_height = std::min(image1.height, image2.height);
            
            image1 = resize_image(image1, target_width, target_height);
            image2 = resize_image(image2, target_width, target_height);
        }
        
        // Compare using Algorithm 3
        result.distance = compare_images(image1, image2, num_threads);
        
        // Store metadata
        result.label_name = image_path1 + " vs " + image_path2;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error in load_and_compare_images: " + std::string(e.what()));
    }
    
    return result;
}

std::vector<DistanceResult> batch_compare_images(
    const Image& reference_image,
    const std::vector<std::string>& image_paths,
    int num_threads
) {
    std::vector<DistanceResult> results;
    results.reserve(image_paths.size());
    
    // Load reference image once
    Image ref_image = reference_image;
    
    // Parallel batch comparison
    std::vector<std::thread> threads;
    std::vector<DistanceResult> thread_results(image_paths.size());
    std::atomic<size_t> current_index(0);
    
    auto worker = [&](int thread_id) {
        while (true) {
            size_t idx = current_index.fetch_add(1);
            if (idx >= image_paths.size()) break;
            
            try {
                Image current_image = load_image(image_paths[idx]);
                
                // Resize to match reference
                if (current_image.width != ref_image.width || 
                    current_image.height != ref_image.height) {
                    current_image = resize_image(current_image, 
                                                ref_image.width, 
                                                ref_image.height);
                }
                
                // Use num_threads=1 for sequential comparison within each parallel worker
                // to avoid thread oversubscription (outer parallelism is across images)
                thread_results[idx].distance = compare_images(ref_image, current_image, 1);
                thread_results[idx].label_name = image_paths[idx];
                
            } catch (const std::exception& e) {
                thread_results[idx].distance = -1.0;  // Error marker
                thread_results[idx].label_name = "Error: " + std::string(e.what());
            }
        }
    };
    
    // Launch threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    return thread_results;
}

bool save_image(const Image& image, const std::string& output_path) {
    try {
        cv::Mat mat = pixel_array_to_mat(image.data, image.channels);
        if (mat.empty()) {
            return false;
        }
        
        if (image.channels == 3) {
            // Convert RGB to BGR for OpenCV saving
            cv::Mat bgr_mat;
            cv::cvtColor(mat, bgr_mat, cv::COLOR_RGB2BGR);
            return cv::imwrite(output_path, bgr_mat);
        } else {
            // Grayscale or single-channel image can be saved directly
            return cv::imwrite(output_path, mat);
        }
    } catch (const std::exception&) {
        return false;
    }
}

std::string get_image_info(const std::string& image_path) {
    cv::Mat mat = cv::imread(image_path, cv::IMREAD_COLOR);
    
    if (mat.empty()) {
        return "Failed to load image: " + image_path;
    }
    
    std::ostringstream oss;
    oss << "Image: " << image_path << "\n";
    oss << "Dimensions: " << mat.cols << "x" << mat.rows << "\n";
    oss << "Channels: " << mat.channels() << "\n";
    oss << "Depth: " << mat.depth() << "\n";
    oss << "Total pixels: " << (mat.cols * mat.rows) << "\n";
    oss << "Size in bytes: " << mat.total() * mat.elemSize();
    
    return oss.str();
}

} // namespace algorithm4
} // namespace cis6007

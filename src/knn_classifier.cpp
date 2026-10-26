#include "knn_classifier.hpp"
#include <fstream>
#include <algorithm>
#include <map>

namespace cis6007 {
namespace knn {

bool load_cifar_batch(const std::string& filename, std::vector<CIFARImage>& images) {
    // Open the binary file
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    // CIFAR-10 batch format:
    // Each image is 3073 bytes: 1 byte label + 3072 bytes pixel data (32x32x3)
    constexpr size_t IMAGE_SIZE = 3073;  // 1 label + 3072 pixels
    constexpr size_t BATCH_SIZE = 10000; // Number of images per batch
    constexpr size_t EXPECTED_FILE_SIZE = BATCH_SIZE * IMAGE_SIZE;

    std::streamsize file_size = file.tellg();
    if (file_size < static_cast<std::streamsize>(EXPECTED_FILE_SIZE)) {
        std::cerr << "Batch file size too small: " << filename << " (" << file_size << " bytes)" << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    // Read the entire batch
    std::vector<uint8_t> buffer(EXPECTED_FILE_SIZE);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    
    if (!file) {
        std::cerr << "Failed to read batch file: " << filename << std::endl;
        return false;
    }
    
    // Parse each image
    for (size_t i = 0; i < BATCH_SIZE; ++i) {
        CIFARImage image;
        size_t offset = i * IMAGE_SIZE;
        
        // First byte is the label
        image.label = buffer[offset];
        
        // Next 3072 bytes are pixel data in channel-major order
        // Format: 1024 R bytes, then 1024 G bytes, then 1024 B bytes
        for (int row = 0; row < 32; ++row) {
            for (int col = 0; col < 32; ++col) {
                size_t pixel_index = row * 32 + col;
                
                // R channel
                image.data[row][col * 3 + 0] = buffer[offset + 1 + pixel_index];
                // G channel
                image.data[row][col * 3 + 1] = buffer[offset + 1 + 1024 + pixel_index];
                // B channel
                image.data[row][col * 3 + 2] = buffer[offset + 1 + 2048 + pixel_index];
            }
        }
        
        images.push_back(image);
    }
    
    file.close();
    return true;
}

PixelArray2D cifar_to_pixel_array(const CIFARImage& cifar_image) {
    return cifar_image.data;
}

bool KNNClassifier::load_dataset(const std::string& data_dir) {
    is_loaded_ = load_training_data(data_dir) && load_test_data(data_dir);
    return is_loaded_;
}

bool KNNClassifier::load_training_data(const std::string& data_dir) {
    training_images_.clear();
    
    // Load 5 training batches: data_batch_1.bin to data_batch_5.bin
    for (int i = 1; i <= 5; ++i) {
        std::string filename = data_dir + "/data_batch_" + std::to_string(i) + ".bin";
        std::vector<CIFARImage> batch_images;
        
        if (!load_cifar_batch(filename, batch_images)) {
            std::cerr << "Failed to load training batch " << i << std::endl;
            return false;
        }
        
        // Append batch images to training set
        training_images_.insert(training_images_.end(), 
                               batch_images.begin(), batch_images.end());
    }
    
    std::cout << "Loaded " << training_images_.size() << " training images" << std::endl;
    return true;
}

bool KNNClassifier::load_test_data(const std::string& data_dir) {
    test_images_.clear();
    
    std::string filename = data_dir + "/test_batch.bin";
    
    if (!load_cifar_batch(filename, test_images_)) {
        std::cerr << "Failed to load test batch" << std::endl;
        return false;
    }
    
    std::cout << "Loaded " << test_images_.size() << " test images" << std::endl;
    return true;
}

std::vector<std::pair<double, size_t>> KNNClassifier::find_k_nearest(const CIFARImage& test_image) {
    if (training_images_.empty()) {
        throw std::runtime_error("Training data not loaded");
    }

    // Convert test image to pixel array
    PixelArray2D test_array = cifar_to_pixel_array(test_image);

    // Compute distances to all training images in parallel
    std::vector<double> distances(training_images_.size());
    std::atomic<size_t> current_index(0);
    int threads_to_use = std::max(1, std::min(num_threads_, static_cast<int>(training_images_.size())));
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);

    auto worker = [&]() {
        while (true) {
            size_t idx = current_index.fetch_add(1);
            if (idx >= training_images_.size()) break;

            // Convert training image to pixel array
            PixelArray2D train_array = cifar_to_pixel_array(training_images_[idx]);

            // Compute percentage distance using Algorithm 5
            try {
                distances[idx] = algorithm5::percentage_distance_parallel(
                    test_array, train_array, 1
                );
            } catch (const std::exception& e) {
                distances[idx] = std::numeric_limits<double>::max();
            }
        }
    };

    // Launch worker threads
    for (int t = 0; t < threads_to_use; ++t) {
        threads.emplace_back(worker);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Create vector of (distance, index) pairs
    std::vector<std::pair<double, size_t>> distance_pairs;
    distance_pairs.reserve(training_images_.size());
    for (size_t i = 0; i < distances.size(); ++i) {
        distance_pairs.emplace_back(distances[i], i);
    }

    size_t effective_k = std::min(static_cast<size_t>(std::max(0, k_)), distance_pairs.size());
    if (effective_k == 0) {
        return {};
    }

    // Find k smallest distances using nth_element + sort
    if (effective_k < distance_pairs.size()) {
        std::nth_element(distance_pairs.begin(),
                         distance_pairs.begin() + effective_k,
                         distance_pairs.end(),
                         [](const auto& a, const auto& b) {
                             return a.first < b.first;
                         });
        std::sort(distance_pairs.begin(),
                  distance_pairs.begin() + effective_k,
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
    } else {
        std::sort(distance_pairs.begin(),
                  distance_pairs.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
    }

    // Return k nearest (distance, index) pairs
    std::vector<std::pair<double, size_t>> nearest;
    nearest.reserve(effective_k);
    for (size_t i = 0; i < effective_k; ++i) {
        nearest.push_back(distance_pairs[i]);
    }

    return nearest;
}

int majority_vote(const std::vector<int>& labels) {
    if (labels.empty()) {
        return -1;
    }
    
    // Count occurrences of each label
    std::map<int, int> label_counts;
    for (int label : labels) {
        label_counts[label]++;
    }
    
    // Find label with maximum count
    int max_count = 0;
    int majority_label = labels[0];
    
    for (const auto& pair : label_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            majority_label = pair.first;
        }
    }
    
    return majority_label;
}

double calculate_confidence(
    const std::vector<double>& distances,
    const std::vector<int>& labels,
    int predicted_label
) {
    if (distances.empty() || labels.empty()) {
        return 0.0;
    }

    // Assignment formula: Confidence = 1 - [Min(PDV) / (Sum(PDV) * N° Training Elements)]
    // Where:
    //   Min(PDV) = minimum percentage distance among k neighbors
    //   Sum(PDV) = sum of percentage distances among k neighbors
    //   N° Training Elements = k (number of neighbors)

    // Find minimum and sum of distances
    double min_distance = distances[0];
    double sum_distances = 0.0;
    for (double dist : distances) {
        min_distance = std::min(min_distance, dist);
        sum_distances += dist;
    }

    // Compute confidence using the assignment formula
    double k = static_cast<double>(distances.size());
    double confidence;

    if (sum_distances * k == 0.0) {
        // Avoid division by zero - all distances are zero means perfect match
        confidence = 1.0;
    } else {
        confidence = 1.0 - (min_distance / (sum_distances * k));
    }

    // Clamp to valid range [0, 1] and convert to percentage
    confidence = std::max(0.0, std::min(1.0, confidence));
    return confidence * 100.0;
}

DistanceResult KNNClassifier::classify(const CIFARImage& test_image) {
    DistanceResult result;

    // Find k nearest neighbors - returns (distance, index) pairs already sorted by distance
    std::vector<std::pair<double, size_t>> nearest = find_k_nearest(test_image);

    if (nearest.empty()) {
        result.distance = -1.0;
        result.confidence = 0.0;
        result.label = -1;
        return result;
    }

    // Extract labels and distances from nearest neighbors (distances already computed)
    std::vector<int> neighbor_labels;
    std::vector<double> neighbor_distances;
    neighbor_labels.reserve(k_);
    neighbor_distances.reserve(k_);

    for (const auto& [dist, idx] : nearest) {
        neighbor_labels.push_back(training_images_[idx].label);
        neighbor_distances.push_back(dist);
    }

    // Majority vote to get predicted label
    result.label = majority_vote(neighbor_labels);
    if (result.label >= 0 && static_cast<size_t>(result.label) < CIFAR_CLASS_NAMES.size()) {
        result.label_name = CIFAR_CLASS_NAMES[result.label];
    } else {
        result.label_name = "unknown";
    }

    // Calculate confidence using the assignment formula
    result.confidence = calculate_confidence(neighbor_distances, neighbor_labels, result.label);

    // Store average distance as the distance metric
    double avg_distance = 0.0;
    for (double dist : neighbor_distances) {
        avg_distance += dist;
    }
    avg_distance /= neighbor_distances.size();
    result.distance = avg_distance;

    return result;
}

DistanceResult KNNClassifier::classify_by_index(size_t test_index) {
    if (test_index >= test_images_.size()) {
        throw std::out_of_range("Test index out of range");
    }
    
    return classify(test_images_[test_index]);
}

std::vector<DistanceResult> KNNClassifier::classify_all() {
    std::vector<DistanceResult> results;
    results.reserve(test_images_.size());

    // Classify each test image sequentially
    for (size_t i = 0; i < test_images_.size(); ++i) {
        results.push_back(classify_by_index(i));

        // Print progress
        if ((i + 1) % 100 == 0) {
            std::cout << "Classified " << (i + 1) << "/" << test_images_.size()
                      << " test images" << std::endl;
        }
    }

    return results;
}

std::vector<DistanceResult> KNNClassifier::classify_all_parallel(int outer_threads) {
    std::vector<DistanceResult> results(test_images_.size());
    std::atomic<size_t> completed_count(0);
    std::vector<std::thread> threads;
    threads.reserve(outer_threads);

    // Worker function that processes test images in parallel
    auto worker = [&](int thread_id) {
        for (size_t i = thread_id; i < test_images_.size(); i += outer_threads) {
            results[i] = classify(test_images_[i]);

            // Atomically increment and print progress
            size_t completed = completed_count.fetch_add(1) + 1;
            if (completed % 100 == 0) {
                std::cout << "Classified " << completed << "/" << test_images_.size()
                          << " test images" << std::endl;
            }
        }
    };

    // Launch worker threads
    for (int t = 0; t < outer_threads; ++t) {
        threads.emplace_back(worker, t);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    return results;
}

double KNNClassifier::compute_accuracy() {
    std::vector<DistanceResult> results = classify_all();
    if (results.empty()) {
        return 0.0;
    }
    
    int correct = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].label == test_images_[i].label) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / results.size() * 100.0;
}

} // namespace knn
} // namespace cis6007

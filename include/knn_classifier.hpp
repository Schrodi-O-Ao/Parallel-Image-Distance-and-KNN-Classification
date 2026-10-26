#ifndef KNN_CLASSIFIER_HPP
#define KNN_CLASSIFIER_HPP

/**
 * @file knn_classifier.hpp
 * @brief K-Nearest Neighbors (KNN) classifier for CIFAR-10 image classification.
 * @author ST20290339
 */

#include "common.hpp"
#include "algorithm5.hpp"

namespace cis6007 {
namespace knn {

/**
 * @class KNNClassifier
 * @brief K-Nearest Neighbors classifier for the CIFAR-10 dataset.
 *
 * Implements parallel image classification across CIFAR-10 training images
 * using Algorithm 5 percentage distance metrics and multithreaded majority voting.
 */
class KNNClassifier {
private:
    std::vector<CIFARImage> training_images_;   ///< CIFAR-10 training images loaded from batch files
    std::vector<CIFARImage> test_images_;       ///< CIFAR-10 test images loaded from test_batch.bin
    int k_;                                     ///< Number of nearest neighbors evaluated during classification
    int num_threads_;                           ///< Worker thread count for parallel distance computations
    bool is_loaded_;                            ///< Flag indicating whether dataset has been successfully loaded

public:
    /**
     * @brief Construct a KNN classifier.
     *
     * @param k Number of nearest neighbors (default: 5).
     * @param num_threads Number of worker threads for parallel computation.
     */
    KNNClassifier(int k = 5, int num_threads = std::thread::hardware_concurrency())
        : k_(k), num_threads_(num_threads), is_loaded_(false) {}
    
    /**
     * @brief Load CIFAR-10 dataset from binary files in the given directory.
     *
     * @param data_dir Directory containing CIFAR-10 binary data files.
     * @return true if successful, false otherwise.
     */
    bool load_dataset(const std::string& data_dir);
    
    /**
     * @brief Load CIFAR-10 training batch files (data_batch_1.bin through data_batch_5.bin).
     *
     * @param data_dir Directory containing training batch files.
     * @return true if successful.
     */
    bool load_training_data(const std::string& data_dir);
    
    /**
     * @brief Load CIFAR-10 test batch file (test_batch.bin).
     *
     * @param data_dir Directory containing test batch file.
     * @return true if successful.
     */
    bool load_test_data(const std::string& data_dir);
    
    /**
     * @brief Classify a single test image using k nearest neighbors.
     *
     * @param test_image Test image to classify.
     * @return DistanceResult with predicted label, heuristic confidence,
     *         and distance set to the mean percentage distance over the
     *         k nearest training images (not the single nearest distance).
     * @warning Requires load_dataset() to have returned true.
     */
    DistanceResult classify(const CIFARImage& test_image);
    
    /**
     * @brief Classify a test image specified by index within the loaded test set.
     *
     * @param test_index Index of test image in test set.
     * @return DistanceResult with predicted label, heuristic confidence,
     *         and distance set to the mean percentage distance over the
     *         k nearest training images.
     * @warning Requires load_dataset() to have returned true.
     */
    DistanceResult classify_by_index(size_t test_index);
    
    /**
     * @brief Classify all test images sequentially.
     *
     * @return Vector of classification results for each test image.
     * @warning Requires load_dataset() to have returned true.
     */
    std::vector<DistanceResult> classify_all();

    /**
     * @brief Classify all test images in parallel using multi-threaded batching.
     *
     * @param outer_threads Number of worker threads for test image distribution.
     * @return Vector of classification results.
     * @warning Requires load_dataset() to have returned true.
     */
    std::vector<DistanceResult> classify_all_parallel(int outer_threads);

    /**
     * @brief Compute overall classification accuracy on the test set.
     *
     * @return Accuracy as percentage in [0, 100].
     */
    double compute_accuracy();
    
    /**
     * @brief Find the k nearest neighbors for a test image.
     *
     * @param test_image Test image.
     * @return Vector of (distance, index) pairs for k nearest neighbors, sorted by ascending distance.
     */
    std::vector<std::pair<double, size_t>> find_k_nearest(const CIFARImage& test_image);
    
    /**
     * Get training images
     */
    const std::vector<CIFARImage>& get_training_images() const { return training_images_; }
    
    /**
     * Get training images (non-const for testing)
     */
    std::vector<CIFARImage>& get_training_images() { return training_images_; }
    
    /**
     * Get test images
     */
    const std::vector<CIFARImage>& get_test_images() const { return test_images_; }
    
    /**
     * Set k value
     */
    void set_k(int k) { k_ = k; }
    
    /**
     * Get k value
     */
    int get_k() const { return k_; }
    
    /**
     * Check if dataset is loaded
     */
    bool is_loaded() const { return is_loaded_; }
    
    /**
     * Get number of training images
     */
    size_t get_training_size() const { return training_images_.size(); }
    
    /**
     * Get number of test images
     */
    size_t get_test_size() const { return test_images_.size(); }
};

/**
 * @brief Helper function to load a single CIFAR-10 binary batch file.
 *
 * @param filename Path to binary batch file.
 * @param images Output vector to store loaded CIFAR images.
 * @return true if successful, false otherwise.
 */
bool load_cifar_batch(const std::string& filename, std::vector<CIFARImage>& images);

/**
 * @brief Convert a CIFARImage structure to the 2D pixel array format.
 *
 * @param cifar_image CIFAR-10 image structure.
 * @return PixelArray2D representation.
 */
PixelArray2D cifar_to_pixel_array(const CIFARImage& cifar_image);

/**
 * @brief Calculate heuristic confidence score based on neighbor distances and vote distribution.
 *
 * @param distances Distances to k nearest neighbors.
 * @param labels Labels of k nearest neighbors.
 * @param predicted_label Predicted class label.
 * @return Confidence score in [0, 100].
 */
double calculate_confidence(
    const std::vector<double>& distances,
    const std::vector<int>& labels,
    int predicted_label
);

/**
 * @brief Perform majority voting across k nearest neighbor labels.
 *
 * @param labels Class labels of the k nearest neighbors.
 * @return Most frequent class label.
 */
int majority_vote(const std::vector<int>& labels);

} // namespace knn
} // namespace cis6007

#endif // KNN_CLASSIFIER_HPP

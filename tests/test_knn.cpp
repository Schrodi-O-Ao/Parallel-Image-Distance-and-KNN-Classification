#include "../include/knn_classifier.hpp"
#include <iostream>
#include <cassert>

void test_cifar_conversion() {
    std::cout << "Test 1: CIFAR to pixel array conversion" << std::endl;
    
    cis6007::CIFARImage cifar_img;
    cifar_img.label = 3;
    
    // Fill with test data
    for (int row = 0; row < 32; ++row) {
        for (int col = 0; col < 32; ++col) {
            cifar_img.data[row][col * 3 + 0] = 100;  // R
            cifar_img.data[row][col * 3 + 1] = 150;  // G
            cifar_img.data[row][col * 3 + 2] = 200;  // B
        }
    }
    
    cis6007::PixelArray2D pixel_array = cis6007::knn::cifar_to_pixel_array(cifar_img);
    
    assert(pixel_array.size() == 32);
    assert(pixel_array[0].size() == 96);  // 32 * 3 channels
    std::cout << "  Conversion successful: 32x96 array" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_knn_classifier_construction() {
    std::cout << "Test 2: KNN classifier construction" << std::endl;
    
    cis6007::knn::KNNClassifier classifier(5, 4);
    assert(classifier.get_k() == 5);
    assert(!classifier.is_loaded());
    std::cout << "  Classifier created with k=5, threads=4" << std::endl;
    
    classifier.set_k(3);
    assert(classifier.get_k() == 3);
    std::cout << "  k value updated to 3" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_majority_vote() {
    std::cout << "Test 3: Majority voting" << std::endl;
    
    std::vector<int> labels = {1, 1, 2, 3, 1};
    int result = cis6007::knn::majority_vote(labels);
    assert(result == 1);
    std::cout << "  Majority of {1,1,2,3,1} is " << result << " (expected 1)" << std::endl;
    
    labels = {0, 0, 1, 1, 2};
    result = cis6007::knn::majority_vote(labels);
    assert(result == 0 || result == 1);  // Tie - either is acceptable
    std::cout << "  Tie case handled: " << result << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_confidence_calculation() {
    std::cout << "Test 4: Confidence calculation" << std::endl;
    
    std::vector<double> distances = {5.0, 10.0, 15.0};
    std::vector<int> labels = {1, 1, 1};
    
    double confidence = cis6007::knn::calculate_confidence(distances, labels, 1);
    assert(confidence > 0.0 && confidence <= 100.0);
    std::cout << "  Confidence with unanimous vote: " << confidence << "%" << std::endl;
    
    labels = {1, 2, 1};
    confidence = cis6007::knn::calculate_confidence(distances, labels, 1);
    assert(confidence > 0.0 && confidence <= 100.0);
    std::cout << "  Confidence with split vote: " << confidence << "%" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_knn_without_dataset() {
    std::cout << "Test 5: KNN without loaded dataset" << std::endl;
    
    cis6007::knn::KNNClassifier classifier(3, 2);
    
    cis6007::CIFARImage test_img;
    test_img.label = 5;
    
    try {
        cis6007::DistanceResult result = classifier.classify(test_img);
        assert(false);  // Should not reach here
    } catch (const std::runtime_error& e) {
        std::cout << "  Correctly throws error when training data not loaded" << std::endl;
    }
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_knn_with_cifar_data() {
    std::cout << "Test 6: KNN with repository CIFAR-10 data" << std::endl;
    
    cis6007::knn::KNNClassifier classifier(3, 2);

    std::string data_path = "data/cifar-10-batches-bin";
    bool loaded = classifier.load_dataset(data_path);
    if (!loaded) {
        data_path = "../data/cifar-10-batches-bin";
        loaded = classifier.load_dataset(data_path);
    }
    if (!loaded) {
        data_path = "../../data/cifar-10-batches-bin";
        loaded = classifier.load_dataset(data_path);
    }

    if (!loaded) {
        std::cout << "  SKIPPED: CIFAR-10 data not found in data/cifar-10-batches-bin" << std::endl << std::endl;
        return;
    }

    assert(classifier.get_training_images().size() == 50000);
    assert(classifier.get_test_images().size() == 10000);

    const auto& test_img = classifier.get_test_images().front();
    cis6007::DistanceResult result = classifier.classify(test_img);
    assert(result.label >= 0 && result.label < cis6007::CIFAR_CLASSES);
    assert(result.distance >= 0.0 && result.distance <= 100.0);
    std::cout << "  Classified one held-out image: true=" << test_img.label
              << ", predicted=" << result.label << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_class_names() {
    std::cout << "Test 7: CIFAR-10 class names" << std::endl;
    
    assert(cis6007::CIFAR_CLASS_NAMES.size() == 10);
    assert(cis6007::CIFAR_CLASS_NAMES[0] == "airplane");
    assert(cis6007::CIFAR_CLASS_NAMES[9] == "truck");
    
    std::cout << "  Class names verified:" << std::endl;
    for (size_t i = 0; i < cis6007::CIFAR_CLASS_NAMES.size(); ++i) {
        std::cout << "    " << i << ": " << cis6007::CIFAR_CLASS_NAMES[i] << std::endl;
    }
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_edge_cases() {
    std::cout << "Test 8: Edge cases" << std::endl;
    
    // Empty labels
    std::vector<int> empty_labels;
    int result = cis6007::knn::majority_vote(empty_labels);
    assert(result == -1);
    std::cout << "  Empty labels handled correctly" << std::endl;
    
    // Single label
    std::vector<int> single_label = {5};
    result = cis6007::knn::majority_vote(single_label);
    assert(result == 5);
    std::cout << "  Single label handled correctly" << std::endl;
    
    // Empty distances
    std::vector<double> empty_distances;
    double confidence = cis6007::knn::calculate_confidence(empty_distances, empty_labels, 0);
    assert(confidence == 0.0);
    std::cout << "  Empty distances handled correctly" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "KNN Classifier Test Suite" << std::endl;
    std::cout << "CIFAR-10 Classification" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    try {
        test_cifar_conversion();
        test_knn_classifier_construction();
        test_majority_vote();
        test_confidence_calculation();
        test_knn_without_dataset();
        test_knn_with_cifar_data();
        test_class_names();
        test_edge_cases();
        
        std::cout << "========================================" << std::endl;
        std::cout << "ALL TESTS PASSED!" << std::endl;
        std::cout << "Note: Full KNN testing requires CIFAR-10 dataset files" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}

#include "common.hpp"
#include "knn_classifier.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace cis6007;

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "CIS6007 - KNN Classification on CIFAR-10\n";
    std::cout << "Student ID: ST20290339\n";
    std::cout << "========================================\n\n";

    if (argc < 2) {
        std::cout << "Usage: knn_demo <cifar_data_directory>\n";
        std::cout << "Example: knn_demo data/cifar-10-batches-bin\n\n";
        return 1;
    }

    std::string data_path = argv[1];
    
    try {
        // Create KNN classifier
        knn::KNNClassifier classifier(5, 4);
        
        std::cout << "Loading CIFAR-10 dataset from:\n";
        std::cout << "  " << data_path << "\n\n";
        
        // Load training data
        std::cout << "Loading training data...\n";
        classifier.load_training_data(data_path);
        std::cout << "Training images loaded: " << classifier.get_training_images().size() << "\n\n";
        
        // Load test data
        std::cout << "Loading test data...\n";
        classifier.load_test_data(data_path);
        std::cout << "Test images loaded: " << classifier.get_test_images().size() << "\n\n";
        
        // Run classification on first 150 test images for meaningful accuracy
        std::cout << "========================================\n";
        std::cout << "Classifying first 150 test images\n";
        std::cout << "========================================\n\n";
        
        const auto& test_images = classifier.get_test_images();
        int num_to_classify = std::min(150, static_cast<int>(test_images.size()));
        
        int correct = 0;
        const auto classification_start = std::chrono::steady_clock::now();
        
        for (int i = 0; i < num_to_classify; ++i) {
            const auto& test_img = test_images[i];
            
            try {
                DistanceResult result = classifier.classify(test_img);
                
                std::string true_name = (test_img.label >= 0 && static_cast<size_t>(test_img.label) < CIFAR_CLASS_NAMES.size())
                    ? CIFAR_CLASS_NAMES[test_img.label] : "unknown";
                std::cout << "Image " << (i + 1) << ":\n";
                std::cout << "  True label: " << test_img.label << " (" << true_name << ")\n";
                std::cout << "  Predicted: " << result.label << " (" << result.label_name << ")\n";
                std::cout << "  Confidence: " << std::fixed << std::setprecision(2) << result.confidence << "%\n";
                std::cout << "  RMS pixel difference: " << std::fixed << std::setprecision(4)
                          << result.distance << "% of the 8-bit channel range\n";
                
                if (result.label == test_img.label) {
                    std::cout << "  Status: CORRECT\n";
                    correct++;
                } else {
                    std::cout << "  Status: INCORRECT\n";
                }
                std::cout << "\n";
                
            } catch (const std::exception& e) {
                std::cerr << "Error classifying image " << i << ": " << e.what() << "\n\n";
            }
        }
        const auto classification_end = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration<double, std::milli>(
            classification_end - classification_start
        ).count();
        
        std::cout << "========================================\n";
        std::cout << "Demo Results\n";
        std::cout << "========================================\n";
        std::cout << "Classified: " << num_to_classify << " images\n";
        std::cout << "Correct: " << correct << "\n";
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) 
                  << (static_cast<double>(correct) / num_to_classify * 100.0) << "%\n\n";
        std::cout << "Classification runtime: " << std::fixed << std::setprecision(2)
                  << elapsed_ms << " ms for " << num_to_classify << " test images\n\n";
        
        std::cout << "Note: This is a demo on 150 images for meaningful accuracy.\n";
        std::cout << "To classify all 10,000 test images, modify the num_to_classify variable.\n";
        std::cout << "Full classification may take several minutes.\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

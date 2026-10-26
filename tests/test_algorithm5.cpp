#include "../include/algorithm5.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

void test_percentage_distance_sequential() {
    std::cout << "Test 1: Sequential percentage distance" << std::endl;
    
    cis6007::PixelArray2D array1 = {{100, 150}, {200, 50}};
    cis6007::PixelArray2D array2 = {{100, 150}, {200, 50}};
    
    double result = cis6007::algorithm5::percentage_distance_sequential(array1, array2);
    assert(result == 0.0);
    std::cout << "  Identical arrays: " << result << " (expected 0.0)" << std::endl;
    
    array2 = {{0, 0}, {0, 0}};
    result = cis6007::algorithm5::percentage_distance_sequential(array1, array2);
    assert(result > 0.0);
    std::cout << "  Different arrays: " << result << " (expected > 0.0)" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_percentage_distance_parallel() {
    std::cout << "Test 2: Parallel percentage distance" << std::endl;
    
    cis6007::PixelArray2D array1 = {{100, 150, 200}, {50, 75, 100}};
    cis6007::PixelArray2D array2 = {{100, 150, 200}, {50, 75, 100}};
    
    double result = cis6007::algorithm5::percentage_distance_parallel(array1, array2, 2);
    assert(result == 0.0);
    std::cout << "  Identical arrays (2 threads): " << result << " (expected 0.0)" << std::endl;
    
    array2 = {{0, 0, 0}, {0, 0, 0}};
    double sequential = cis6007::algorithm5::percentage_distance_sequential(array1, array2);
    double parallel = cis6007::algorithm5::percentage_distance_parallel(array1, array2, 4);
    assert(std::abs(sequential - parallel) < 1e-6);
    std::cout << "  Parallel vs sequential match: " << parallel << " vs " << sequential << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_percentage_distance_images() {
    std::cout << "Test 3: Percentage distance for Image structures" << std::endl;
    
    cis6007::Image img1(50, 50, 3);
    cis6007::Image img2(50, 50, 3);
    
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 150; ++j) {
            img1.data[i][j] = 128;
            img2.data[i][j] = 128;
        }
    }
    
    double result = cis6007::algorithm5::percentage_distance_images(img1, img2, 2);
    assert(result == 0.0);
    std::cout << "  Identical images: " << result << " (expected 0.0)" << std::endl;
    
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 150; ++j) {
            img2.data[i][j] = 0;
        }
    }
    
    result = cis6007::algorithm5::percentage_distance_images(img1, img2, 2);
    assert(result > 0.0);
    std::cout << "  Different images: " << result << " (expected > 0.0)" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_percentage_distance_detailed() {
    std::cout << "Test 4: Detailed percentage distance" << std::endl;
    
    cis6007::PixelArray2D array1 = {{100, 150}, {200, 50}};
    cis6007::PixelArray2D array2 = {{100, 150}, {200, 50}};
    
    cis6007::DistanceResult result = cis6007::algorithm5::percentage_distance_detailed(array1, array2, 2);
    assert(result.distance == 0.0);
    assert(result.confidence == 100.0);
    std::cout << "  Identical arrays - Distance: " << result.distance << ", Confidence: " << result.confidence << std::endl;
    
    array2 = {{0, 0}, {0, 0}};
    result = cis6007::algorithm5::percentage_distance_detailed(array1, array2, 2);
    assert(result.distance > 0.0);
    assert(result.confidence < 100.0);
    std::cout << "  Different arrays - Distance: " << result.distance << ", Confidence: " << result.confidence << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_percentage_to_similarity() {
    std::cout << "Test 5: Percentage to similarity conversion" << std::endl;
    
    double similarity = cis6007::algorithm5::percentage_to_similarity(0.0);
    assert(similarity == 100.0);
    std::cout << "  Zero distance -> 100% similarity" << std::endl;
    
    similarity = cis6007::algorithm5::percentage_to_similarity(50.0);
    assert(similarity > 0.0 && similarity < 100.0);
    std::cout << "  50% distance -> " << similarity << "% similarity" << std::endl;
    
    similarity = cis6007::algorithm5::percentage_to_similarity(500.0);
    assert(similarity >= 0.0);
    std::cout << "  500% distance -> " << similarity << "% similarity (clamped)" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_interpret_percentage_distance() {
    std::cout << "Test 6: Interpret percentage distance" << std::endl;
    
    std::string interpretation = cis6007::algorithm5::interpret_percentage_distance(2.0);
    assert(interpretation.find("Very similar") != std::string::npos);
    std::cout << "  2.0%: " << interpretation << std::endl;
    
    interpretation = cis6007::algorithm5::interpret_percentage_distance(10.0);
    assert(interpretation.find("Similar") != std::string::npos);
    std::cout << "  10.0%: " << interpretation << std::endl;
    
    interpretation = cis6007::algorithm5::interpret_percentage_distance(40.0);
    assert(interpretation.find("Different") != std::string::npos);
    std::cout << "  40.0%: " << interpretation << std::endl;
    
    interpretation = cis6007::algorithm5::interpret_percentage_distance(150.0);
    assert(interpretation.find("Extremely different") != std::string::npos);
    std::cout << "  150.0%: " << interpretation << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_batch_percentage_distance() {
    std::cout << "Test 7: Batch percentage distance" << std::endl;
    
    cis6007::PixelArray2D reference = {{100, 150}, {200, 50}};
    std::vector<cis6007::PixelArray2D> comparisons = {
        {{100, 150}, {200, 50}},
        {{0, 0}, {0, 0}},
        {{50, 75}, {100, 25}}
    };
    
    std::vector<double> results = cis6007::algorithm5::batch_percentage_distance(reference, comparisons, 2);
    assert(results.size() == 3);
    assert(results[0] == 0.0);
    assert(results[1] > 0.0);
    assert(results[2] > 0.0);
    
    std::cout << "  Batch comparison results: ";
    for (double r : results) {
        std::cout << r << " ";
    }
    std::cout << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_normalization() {
    std::cout << "Test 8: Normalization correctness" << std::endl;
    
    // Repeating the same per-channel difference across more pixels must not change RMS distance.
    cis6007::PixelArray2D array1 = {{10, 10}, {10, 10}};
    cis6007::PixelArray2D array2 = {{0, 0}, {0, 0}};
    
    double result = cis6007::algorithm5::percentage_distance_sequential(array1, array2);
    assert(std::abs(result - (10.0 / 255.0 * 100.0)) < 1e-6);

    cis6007::PixelArray2D larger1(4, std::vector<cis6007::Pixel>(2, 10));
    cis6007::PixelArray2D larger2(4, std::vector<cis6007::Pixel>(2, 0));
    double larger_result = cis6007::algorithm5::percentage_distance_parallel(larger1, larger2, 2);
    assert(std::abs(result - larger_result) < 1e-6);

    cis6007::Image image1(1, 1, 3);
    cis6007::Image image2(1, 1, 3);
    std::fill(image1.data[0].begin(), image1.data[0].end(), 10);
    std::fill(image2.data[0].begin(), image2.data[0].end(), 0);
    double image_result = cis6007::algorithm5::percentage_distance_images(image1, image2, 1);
    assert(std::abs(result - image_result) < 1e-6);

    std::cout << "  RMS percentage: " << result << "%; repeated dimensions and image overload agree" << std::endl;
    std::cout << "  PASSED" << std::endl << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Algorithm 5 Test Suite" << std::endl;
    std::cout << "Percentage Distance" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    try {
        test_percentage_distance_sequential();
        test_percentage_distance_parallel();
        test_percentage_distance_images();
        test_percentage_distance_detailed();
        test_percentage_to_similarity();
        test_interpret_percentage_distance();
        test_batch_percentage_distance();
        test_normalization();
        
        std::cout << "========================================" << std::endl;
        std::cout << "ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}

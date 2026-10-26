#include "../include/algorithm4.hpp"
#include <iostream>
#include <cassert>
#include <fstream>

void test_mat_conversion() {
    std::cout << "Test 1: Mat to PixelArray conversion" << std::endl;
    
    // Create a simple test image
    cv::Mat test_mat(3, 3, CV_8UC3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            test_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(100, 150, 200);
        }
    }
    
    cis6007::PixelArray2D array = cis6007::algorithm4::mat_to_pixel_array(test_mat);
    
    assert(array.size() == 3);
    assert(array[0].size() == 9);  // 3 pixels * 3 channels
    std::cout << "  Conversion successful: " << array.size() << "x" << array[0].size() << std::endl;
    
    cv::Mat reconstructed = cis6007::algorithm4::pixel_array_to_mat(array, 3);
    assert(reconstructed.rows == 3);
    assert(reconstructed.cols == 3);
    std::cout << "  Reconstruction successful" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_image_structure() {
    std::cout << "Test 2: Image structure" << std::endl;
    
    cis6007::Image img(100, 100, 3);
    assert(img.width == 100);
    assert(img.height == 100);
    assert(img.channels == 3);
    assert(img.data.size() == 100);
    std::cout << "  Image structure created correctly" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_resize_image() {
    std::cout << "Test 3: Image resizing" << std::endl;
    
    cis6007::Image img(100, 100, 3);
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 300; ++j) {
            img.data[i][j] = 128;
        }
    }
    
    cis6007::Image resized = cis6007::algorithm4::resize_image(img, 50, 50);
    assert(resized.width == 50);
    assert(resized.height == 50);
    assert(resized.channels == 3);
    std::cout << "  Image resized from 100x100 to 50x50" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_grayscale_conversion() {
    std::cout << "Test 4: Grayscale conversion" << std::endl;
    
    cis6007::Image img(50, 50, 3);
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 150; ++j) {
            img.data[i][j] = 128;
        }
    }
    
    cis6007::Image gray = cis6007::algorithm4::convert_to_grayscale(img);
    assert(gray.channels == 1);
    assert(gray.width == 50);
    assert(gray.height == 50);
    std::cout << "  Converted to grayscale: " << gray.channels << " channel" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_image_comparison() {
    std::cout << "Test 5: Image comparison" << std::endl;
    
    cis6007::Image img1(50, 50, 3);
    cis6007::Image img2(50, 50, 3);
    
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 150; ++j) {
            img1.data[i][j] = 128;
            img2.data[i][j] = 128;
        }
    }
    
    double distance = cis6007::algorithm4::compare_images(img1, img2, 2);
    assert(distance == 0.0);
    std::cout << "  Identical images distance: " << distance << " (expected 0.0)" << std::endl;
    
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 150; ++j) {
            img2.data[i][j] = 0;
        }
    }
    
    distance = cis6007::algorithm4::compare_images(img1, img2, 2);
    assert(distance > 0.0);
    std::cout << "  Different images distance: " << distance << " (expected > 0.0)" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_edge_cases() {
    std::cout << "Test 6: Edge cases" << std::endl;
    
    cis6007::Image img1(10, 10, 3);
    cis6007::Image img2(20, 20, 3);
    
    try {
        cis6007::algorithm4::compare_images(img1, img2, 1);
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "  Dimension mismatch correctly detected" << std::endl;
    }
    
    cis6007::Image img3(10, 10, 3);
    cis6007::Image img4(10, 10, 1);
    
    try {
        cis6007::algorithm4::compare_images(img3, img4, 1);
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "  Channel mismatch correctly detected" << std::endl;
    }
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Algorithm 4 Test Suite" << std::endl;
    std::cout << "Image Loading with OpenCV" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    try {
        test_mat_conversion();
        test_image_structure();
        test_resize_image();
        test_grayscale_conversion();
        test_image_comparison();
        test_edge_cases();
        
        std::cout << "========================================" << std::endl;
        std::cout << "ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}

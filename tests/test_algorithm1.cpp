#include "../include/algorithm1.hpp"
#include <iostream>
#include <cassert>
#include <random>

void test_pixel_difference_basic() {
    std::cout << "Test 1: Basic pixel difference" << std::endl;
    
    // Test with identical pixels
    double result = cis6007::algorithm1::pixel_difference(100, 100);
    assert(result == 0.0);
    std::cout << "  Identical pixels: " << result << " (expected 0.0)" << std::endl;
    
    // Test with different pixels
    result = cis6007::algorithm1::pixel_difference(200, 100);
    assert(result == 10000.0);  // (200-100)² = 10000
    std::cout << "  Different pixels: " << result << " (expected 10000.0)" << std::endl;
    
    // Test with extreme values
    result = cis6007::algorithm1::pixel_difference(255, 0);
    assert(result == 65025.0);  // (255-0)² = 65025
    std::cout << "  Extreme values: " << result << " (expected 65025.0)" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_pixel_difference_int() {
    std::cout << "Test 2: Integer pixel difference" << std::endl;
    
    int result = cis6007::algorithm1::pixel_difference_int(150, 50);
    assert(result == 10000);  // (150-50)² = 10000
    std::cout << "  Integer difference: " << result << " (expected 10000)" << std::endl;
    
    result = cis6007::algorithm1::pixel_difference_int(0, 255);
    assert(result == 65025);
    std::cout << "  Reverse order: " << result << " (expected 65025)" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_pixel_absolute_difference() {
    std::cout << "Test 3: Absolute pixel difference" << std::endl;
    
    double result = cis6007::algorithm1::pixel_absolute_difference(200, 100);
    assert(result == 100.0);
    std::cout << "  Absolute difference: " << result << " (expected 100.0)" << std::endl;
    
    result = cis6007::algorithm1::pixel_absolute_difference(100, 200);
    assert(result == 100.0);  // Should be symmetric
    std::cout << "  Symmetry test: " << result << " (expected 100.0)" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_pixel_difference_normalized() {
    std::cout << "Test 4: Normalized pixel difference" << std::endl;
    
    double result = cis6007::algorithm1::pixel_difference_normalized(255, 0);
    assert(result == 1.0);  // Maximum difference normalized to 1
    std::cout << "  Max difference normalized: " << result << " (expected 1.0)" << std::endl;
    
    result = cis6007::algorithm1::pixel_difference_normalized(128, 127);
    double expected = 1.0 / 65025.0;  // (128-127)² / 255²
    assert(std::abs(result - expected) < 1e-10);
    std::cout << "  Small difference normalized: " << result << " (expected " << expected << ")" << std::endl;
    
    result = cis6007::algorithm1::pixel_difference_normalized(100, 100);
    assert(result == 0.0);
    std::cout << "  Identical pixels normalized: " << result << " (expected 0.0)" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_random_pixels() {
    std::cout << "Test 5: Random pixel values" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    for (int i = 0; i < 10; ++i) {
        cis6007::Pixel x = dis(gen);
        cis6007::Pixel y = dis(gen);
        
        double diff = cis6007::algorithm1::pixel_difference(x, y);
        int diff_int = cis6007::algorithm1::pixel_difference_int(x, y);
        
        // Verify integer and double versions match
        assert(std::abs(diff - diff_int) < 1e-6);
        
        // Verify non-negative
        assert(diff >= 0.0);
        
        // Verify maximum possible value
        assert(diff <= 65025.0);
    }
    
    std::cout << "  10 random pixel pairs tested successfully" << std::endl;
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_symmetry_property() {
    std::cout << "Test 6: Symmetry property" << std::endl;
    
    // pixel_difference should be symmetric: diff(x,y) == diff(y,x)
    for (cis6007::Pixel x = 0; x <= 255; x += 51) {
        for (cis6007::Pixel y = 0; y <= 255; y += 51) {
            double diff_xy = cis6007::algorithm1::pixel_difference(x, y);
            double diff_yx = cis6007::algorithm1::pixel_difference(y, x);
            assert(diff_xy == diff_yx);
        }
    }
    
    std::cout << "  Symmetry verified for sample pixel values" << std::endl;
    std::cout << "  PASSED" << std::endl << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Algorithm 1 Test Suite" << std::endl;
    std::cout << "Pixel Difference Computation" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    try {
        test_pixel_difference_basic();
        test_pixel_difference_int();
        test_pixel_absolute_difference();
        test_pixel_difference_normalized();
        test_random_pixels();
        test_symmetry_property();
        
        std::cout << "========================================" << std::endl;
        std::cout << "ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}

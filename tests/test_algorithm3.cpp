#include "../include/algorithm3.hpp"
#include <iostream>
#include <cassert>
#include <random>

void test_2d_distance_sequential() {
    std::cout << "Test 1: Sequential 2D Euclidean distance" << std::endl;
    
    cis6007::PixelArray2D array1 = {{100, 150}, {200, 50}};
    cis6007::PixelArray2D array2 = {{100, 150}, {200, 50}};
    
    double result = cis6007::algorithm3::euclidean_distance_2d_sequential(array1, array2);
    assert(result == 0.0);
    std::cout << "  Identical 2D arrays: " << result << " (expected 0.0)" << std::endl;
    
    array2 = {{0, 0}, {0, 0}};
    result = cis6007::algorithm3::euclidean_distance_2d_sequential(array1, array2);
    double expected = std::sqrt(100.0*100.0 + 150.0*150.0 + 200.0*200.0 + 50.0*50.0);
    assert(std::abs(result - expected) < 1e-6);
    std::cout << "  Against zeros: " << result << " (expected " << expected << ")" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_2d_distance_parallel() {
    std::cout << "Test 2: Parallel 2D Euclidean distance" << std::endl;
    
    cis6007::PixelArray2D array1 = {{100, 150, 200}, {50, 75, 100}};
    cis6007::PixelArray2D array2 = {{100, 150, 200}, {50, 75, 100}};
    
    double result = cis6007::algorithm3::euclidean_distance_2d_parallel(array1, array2, 2);
    assert(result == 0.0);
    std::cout << "  Identical arrays (2 threads): " << result << " (expected 0.0)" << std::endl;
    
    array2 = {{0, 0, 0}, {0, 0, 0}};
    double sequential = cis6007::algorithm3::euclidean_distance_2d_sequential(array1, array2);
    double parallel = cis6007::algorithm3::euclidean_distance_2d_parallel(array1, array2, 4);
    assert(std::abs(sequential - parallel) < 1e-6);
    std::cout << "  Parallel vs sequential match: " << parallel << " vs " << sequential << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_2d_distance_large_image() {
    std::cout << "Test 3: Large image parallel computation" << std::endl;
    
    const int height = 100;
    const int width = 100;
    
    cis6007::PixelArray2D array1(height, cis6007::PixelArray1D(width, 128));
    cis6007::PixelArray2D array2(height, cis6007::PixelArray1D(width, 128));
    
    double start = cis6007::utils::get_timestamp_ms();
    double result = cis6007::algorithm3::euclidean_distance_2d_parallel(array1, array2, 8);
    double end = cis6007::utils::get_timestamp_ms();
    
    assert(result == 0.0);
    std::cout << "  Large identical 2D arrays: " << result << " (expected 0.0)" << std::endl;
    std::cout << "  Computation time: " << (end - start) << " ms" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_row_parallel() {
    std::cout << "Test 4: Row-level parallelism" << std::endl;
    
    cis6007::PixelArray2D array1 = {{10, 20}, {30, 40}, {50, 60}};
    cis6007::PixelArray2D array2 = {{5, 15}, {25, 35}, {45, 55}};
    
    double result = cis6007::algorithm3::euclidean_distance_2d_row_parallel(array1, array2, 2);
    double sequential = cis6007::algorithm3::euclidean_distance_2d_sequential(array1, array2);
    assert(std::abs(result - sequential) < 1e-6);
    std::cout << "  Row parallel vs sequential: " << result << " vs " << sequential << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_nested_parallel() {
    std::cout << "Test 5: Nested parallelism" << std::endl;
    
    cis6007::PixelArray2D array1(50, cis6007::PixelArray1D(50, 100));
    cis6007::PixelArray2D array2(50, cis6007::PixelArray1D(50, 150));
    
    double result = cis6007::algorithm3::euclidean_distance_2d_nested_parallel(array1, array2, 4, 2);
    double sequential = cis6007::algorithm3::euclidean_distance_2d_sequential(array1, array2);
    assert(std::abs(result - sequential) < 1e-6);
    std::cout << "  Nested parallel vs sequential: " << result << " vs " << sequential << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_edge_cases() {
    std::cout << "Test 6: Edge cases" << std::endl;
    
    cis6007::PixelArray2D empty1, empty2;
    double result = cis6007::algorithm3::euclidean_distance_2d_sequential(empty1, empty2);
    assert(result == 0.0);
    std::cout << "  Empty 2D arrays: " << result << " (expected 0.0)" << std::endl;
    
    cis6007::PixelArray2D single1 = {{128}};
    cis6007::PixelArray2D single2 = {{128}};
    result = cis6007::algorithm3::euclidean_distance_2d_sequential(single1, single2);
    assert(result == 0.0);
    std::cout << "  Single element identical: " << result << " (expected 0.0)" << std::endl;
    
    try {
        cis6007::PixelArray2D a = {{1, 2}, {3, 4}};
        cis6007::PixelArray2D b = {{1, 2}};
        cis6007::algorithm3::euclidean_distance_2d_sequential(a, b);
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "  Row count mismatch correctly detected" << std::endl;
    }
    
    try {
        cis6007::PixelArray2D a = {{1, 2}, {3, 4}};
        cis6007::PixelArray2D b = {{1}, {3}};
        cis6007::algorithm3::euclidean_distance_2d_sequential(a, b);
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "  Width mismatch correctly detected" << std::endl;
    }
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Algorithm 3 Test Suite" << std::endl;
    std::cout << "2D Parallel Comparison" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    try {
        test_2d_distance_sequential();
        test_2d_distance_parallel();
        test_2d_distance_large_image();
        test_row_parallel();
        test_nested_parallel();
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

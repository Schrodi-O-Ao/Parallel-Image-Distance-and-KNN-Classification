#include "../include/algorithm2.hpp"
#include <iostream>
#include <cassert>
#include <random>

void test_1d_distance_sequential() {
    std::cout << "Test 1: Sequential 1D Euclidean distance" << std::endl;
    
    cis6007::PixelArray1D array1 = {100, 150, 200, 50, 75};
    cis6007::PixelArray1D array2 = {100, 150, 200, 50, 75};
    
    double result = cis6007::algorithm2::euclidean_distance_1d_sequential(array1, array2);
    assert(result == 0.0);
    std::cout << "  Identical arrays: " << result << " (expected 0.0)" << std::endl;
    
    array2 = {0, 0, 0, 0, 0};
    result = cis6007::algorithm2::euclidean_distance_1d_sequential(array1, array2);
    double expected = std::sqrt(100.0*100.0 + 150.0*150.0 + 200.0*200.0 + 50.0*50.0 + 75.0*75.0);
    assert(std::abs(result - expected) < 1e-6);
    std::cout << "  Against zeros: " << result << " (expected " << expected << ")" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_1d_distance_parallel() {
    std::cout << "Test 2: Parallel 1D Euclidean distance" << std::endl;
    
    cis6007::PixelArray1D array1 = {100, 150, 200, 50, 75};
    cis6007::PixelArray1D array2 = {100, 150, 200, 50, 75};
    
    double result = cis6007::algorithm2::euclidean_distance_1d_parallel(array1, array2, 2);
    assert(result == 0.0);
    std::cout << "  Identical arrays (2 threads): " << result << " (expected 0.0)" << std::endl;
    
    array2 = {0, 0, 0, 0, 0};
    double sequential = cis6007::algorithm2::euclidean_distance_1d_sequential(array1, array2);
    double parallel = cis6007::algorithm2::euclidean_distance_1d_parallel(array1, array2, 4);
    assert(std::abs(sequential - parallel) < 1e-6);
    std::cout << "  Parallel vs sequential match: " << parallel << " vs " << sequential << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_1d_distance_large_array() {
    std::cout << "Test 3: Large array parallel computation" << std::endl;
    
    const size_t size = 10000;
    cis6007::PixelArray1D array1(size, 128);
    cis6007::PixelArray1D array2(size, 128);
    
    double start = cis6007::utils::get_timestamp_ms();
    double result = cis6007::algorithm2::euclidean_distance_1d_parallel(array1, array2, 8);
    double end = cis6007::utils::get_timestamp_ms();
    
    assert(result == 0.0);
    std::cout << "  Large identical arrays: " << result << " (expected 0.0)" << std::endl;
    std::cout << "  Computation time: " << (end - start) << " ms" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_sum_squared_differences() {
    std::cout << "Test 4: Sum of squared differences" << std::endl;
    
    cis6007::PixelArray1D array1 = {10, 20, 30};
    cis6007::PixelArray1D array2 = {5, 15, 25};
    
    double result = cis6007::algorithm2::sum_squared_differences_1d_parallel(array1, array2, 2);
    double expected = 25.0 + 25.0 + 25.0;  // (10-5)² + (20-15)² + (30-25)²
    assert(std::abs(result - expected) < 1e-6);
    std::cout << "  Sum of squared differences: " << result << " (expected " << expected << ")" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_different_thread_counts() {
    std::cout << "Test 5: Different thread counts" << std::endl;
    
    cis6007::PixelArray1D array1(1000, 100);
    cis6007::PixelArray1D array2(1000, 150);
    
    double result_1 = cis6007::algorithm2::euclidean_distance_1d_parallel(array1, array2, 1);
    double result_2 = cis6007::algorithm2::euclidean_distance_1d_parallel(array1, array2, 2);
    double result_4 = cis6007::algorithm2::euclidean_distance_1d_parallel(array1, array2, 4);
    double result_8 = cis6007::algorithm2::euclidean_distance_1d_parallel(array1, array2, 8);
    
    // All should produce the same result
    assert(std::abs(result_1 - result_2) < 1e-6);
    assert(std::abs(result_2 - result_4) < 1e-6);
    assert(std::abs(result_4 - result_8) < 1e-6);
    
    std::cout << "  Thread count 1: " << result_1 << std::endl;
    std::cout << "  Thread count 2: " << result_2 << std::endl;
    std::cout << "  Thread count 4: " << result_4 << std::endl;
    std::cout << "  Thread count 8: " << result_8 << std::endl;
    std::cout << "  All results match" << std::endl;
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

void test_edge_cases() {
    std::cout << "Test 6: Edge cases" << std::endl;
    
    cis6007::PixelArray1D empty1, empty2;
    double result = cis6007::algorithm2::euclidean_distance_1d_sequential(empty1, empty2);
    assert(result == 0.0);
    std::cout << "  Empty arrays: " << result << " (expected 0.0)" << std::endl;
    
    cis6007::PixelArray1D single1 = {128};
    cis6007::PixelArray1D single2 = {128};
    result = cis6007::algorithm2::euclidean_distance_1d_sequential(single1, single2);
    assert(result == 0.0);
    std::cout << "  Single element identical: " << result << " (expected 0.0)" << std::endl;
    
    cis6007::PixelArray1D array1 = {255};
    cis6007::PixelArray1D array2 = {0};
    result = cis6007::algorithm2::euclidean_distance_1d_sequential(array1, array2);
    assert(result == 255.0);
    std::cout << "  Single element different: " << result << " (expected 255.0)" << std::endl;
    
    try {
        cis6007::PixelArray1D a = {1, 2, 3};
        cis6007::PixelArray1D b = {1, 2};
        cis6007::algorithm2::euclidean_distance_1d_sequential(a, b);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument& e) {
        std::cout << "  Size mismatch correctly detected" << std::endl;
    }
    
    std::cout << "  PASSED" << std::endl << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Algorithm 2 Test Suite" << std::endl;
    std::cout << "1D Parallel Comparison" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    try {
        test_1d_distance_sequential();
        test_1d_distance_parallel();
        test_1d_distance_large_array();
        test_sum_squared_differences();
        test_different_thread_counts();
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

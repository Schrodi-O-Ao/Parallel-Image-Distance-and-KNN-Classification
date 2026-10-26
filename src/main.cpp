#include "common.hpp"
#include "algorithm1.hpp"
#include "algorithm2.hpp"
#include "algorithm3.hpp"
#include "algorithm4.hpp"
#include "algorithm5.hpp"
#include "knn_classifier.hpp"
#include "parallel_utils.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>

using namespace cis6007;

void print_separator(const std::string& title) {
    std::cout << "\n========================================\n";
    std::cout << title << "\n";
    std::cout << "========================================\n\n";
}

void demo_algorithm1() {
    print_separator("Algorithm 1: Pixel Difference");
    
    std::cout << "Computing (x - y)² for pixel values\n\n";
    
    Pixel x = 200, y = 100;
    double result = algorithm1::pixel_difference(x, y);
    
    std::cout << "Pixel 1 (x): " << static_cast<int>(x) << "\n";
    std::cout << "Pixel 2 (y): " << static_cast<int>(y) << "\n";
    std::cout << "Squared difference: " << result << "\n";
    std::cout << "Formula: (" << static_cast<int>(x) << " - " << static_cast<int>(y) << ")² = " << result << "\n";
    
    std::cout << "\nNormalized version: " << algorithm1::pixel_difference_normalized(x, y) << "\n";
}

void demo_algorithm2() {
    print_separator("Algorithm 2: 1D Parallel Comparison");
    
    std::cout << "Computing Euclidean distance Ed1 for 1D arrays\n\n";
    
    PixelArray1D array1 = {100, 150, 200, 50, 75, 125, 175, 25};
    PixelArray1D array2 = {90, 140, 190, 60, 85, 115, 165, 35};
    
    std::cout << "Array 1: ";
    for (auto p : array1) std::cout << static_cast<int>(p) << " ";
    std::cout << "\nArray 2: ";
    for (auto p : array2) std::cout << static_cast<int>(p) << " ";
    std::cout << "\n\n";
    
    double start = utils::get_timestamp_ms();
    double sequential = algorithm2::euclidean_distance_1d_sequential(array1, array2);
    double end_seq = utils::get_timestamp_ms();
    
    double start_par = utils::get_timestamp_ms();
    double parallel = algorithm2::euclidean_distance_1d_parallel(array1, array2, 4);
    double end_par = utils::get_timestamp_ms();
    
    std::cout << "Sequential Ed1: " << sequential << " (time: " << (end_seq - start) << " ms)\n";
    std::cout << "Parallel Ed1 (4 threads): " << parallel << " (time: " << (end_par - start_par) << " ms)\n";
    std::cout << "Speedup: " << (end_seq - start) / (end_par - start_par) << "x\n";
}

void demo_algorithm3() {
    print_separator("Algorithm 3: 2D Parallel Comparison");
    
    std::cout << "Computing 2D Euclidean distance Ed2 for 2D arrays\n\n";
    
    const int height = 50, width = 50;
    PixelArray2D array1(height, PixelArray1D(width, 128));
    PixelArray2D array2(height, PixelArray1D(width, 130));
    
    std::cout << "Array dimensions: " << height << "x" << width << "\n";
    std::cout << "Array 1: All pixels = 128\n";
    std::cout << "Array 2: All pixels = 130\n\n";
    
    double start = utils::get_timestamp_ms();
    double sequential = algorithm3::euclidean_distance_2d_sequential(array1, array2);
    double end_seq = utils::get_timestamp_ms();
    
    double start_par = utils::get_timestamp_ms();
    double parallel = algorithm3::euclidean_distance_2d_parallel(array1, array2, 8);
    double end_par = utils::get_timestamp_ms();
    
    std::cout << "Sequential Ed2: " << sequential << " (time: " << (end_seq - start) << " ms)\n";
    std::cout << "Parallel Ed2 (8 threads): " << parallel << " (time: " << (end_par - start_par) << " ms)\n";
    std::cout << "Speedup: " << (end_seq - start) / (end_par - start_par) << "x\n";
    
    std::cout << "\nNested parallelism test:\n";
    double start_nested = utils::get_timestamp_ms();
    double nested = algorithm3::euclidean_distance_2d_nested_parallel(array1, array2, 4, 2);
    double end_nested = utils::get_timestamp_ms();
    std::cout << "Nested Ed2 (4 rows, 2 pixel threads): " << nested << " (time: " << (end_nested - start_nested) << " ms)\n";
}

void demo_algorithm4() {
    print_separator("Algorithm 4: Image Loading with OpenCV");
    
    std::cout << "Loading and comparing images using OpenCV\n\n";
    
    // Create synthetic test images
    Image img1(100, 100, 3);
    Image img2(100, 100, 3);
    
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 300; ++j) {
            img1.data[i][j] = 128;
            img2.data[i][j] = (i + j) % 256;
        }
    }
    
    std::cout << "Created synthetic images: 100x100x3\n";
    std::cout << "Image 1: Uniform gray (128)\n";
    std::cout << "Image 2: Gradient pattern\n\n";
    
    double distance = algorithm4::compare_images(img1, img2, 4);
    std::cout << "Euclidean distance between images: " << distance << "\n";
    
    // Test grayscale conversion
    Image gray = algorithm4::convert_to_grayscale(img1);
    std::cout << "\nConverted to grayscale: " << gray.channels << " channel(s)\n";
    
    // Test resize
    Image resized = algorithm4::resize_image(img1, 50, 50);
    std::cout << "Resized to: " << resized.width << "x" << resized.height << "\n";
}

void demo_algorithm5() {
    print_separator("Algorithm 5: Percentage Distance");
    
    std::cout << "Normalizing Ed2 by (N × M) to get percentage distance\n\n";
    
    const int height = 100, width = 100;
    PixelArray2D array1(height, PixelArray1D(width, 128));
    PixelArray2D array2(height, PixelArray1D(width, 138));
    
    std::cout << "Array dimensions: " << height << "x" << width << " (total: " << (height * width) << " pixels)\n";
    std::cout << "Array 1: All pixels = 128\n";
    std::cout << "Array 2: All pixels = 138\n\n";
    
    DistanceResult result = algorithm5::percentage_distance_detailed(array1, array2, 4);
    
    std::cout << "Percentage distance: " << std::fixed << std::setprecision(2) << result.distance << "%\n";
    std::cout << "Confidence score: " << std::fixed << std::setprecision(2) << result.confidence << "%\n";
    std::cout << "Interpretation: " << algorithm5::interpret_percentage_distance(result.distance) << "\n";
    
    std::cout << "\nSimilarity score: " << algorithm5::percentage_to_similarity(result.distance) << "%\n";
}

void demo_parallel_utils() {
    print_separator("Parallel Utilities");
    
    std::cout << "Demonstrating parallel utility functions\n\n";
    
    // Test optimal thread count
    size_t small_workload = 100;
    size_t medium_workload = 5000;
    size_t large_workload = 50000;
    
    std::cout << "Optimal thread counts:\n";
    std::cout << "  Small workload (" << small_workload << "): " 
              << parallel_utils::get_optimal_thread_count(small_workload) << " threads\n";
    std::cout << "  Medium workload (" << medium_workload << "): " 
              << parallel_utils::get_optimal_thread_count(medium_workload) << " threads\n";
    std::cout << "  Large workload (" << large_workload << "): " 
              << parallel_utils::get_optimal_thread_count(large_workload) << " threads\n";
    
    // Test parallel reduction
    std::cout << "\nParallel reduction:\n";
    std::vector<double> values(10000);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = i * 0.1;
    }
    
    double sum = parallel_utils::ParallelReduction<double>::sum(values, 4);
    double min_val = parallel_utils::ParallelReduction<double>::min(values, 4);
    double max_val = parallel_utils::ParallelReduction<double>::max(values, 4);
    
    std::cout << "  Sum: " << sum << "\n";
    std::cout << "  Min: " << min_val << "\n";
    std::cout << "  Max: " << max_val << "\n";
    
    // Test parallel map
    std::cout << "\nParallel map:\n";
    auto squared = parallel_utils::parallel_map(values, [](double x) { return x * x; }, 4);
    std::cout << "  Squared first element: " << squared[0] << "\n";
    std::cout << "  Squared last element: " << squared.back() << "\n";
}

void demo_knn_classifier() {
    print_separator("KNN Classifier for CIFAR-10");
    
    std::cout << "K-Nearest Neighbors classification on CIFAR-10 dataset\n\n";
    
    knn::KNNClassifier classifier(5, 4);
    
    std::cout << "Created KNN classifier with:\n";
    std::cout << "  k = " << classifier.get_k() << " (nearest neighbors)\n";
    std::cout << "  threads = 4\n\n";
    
    std::cout << "CIFAR-10 classes:\n";
    for (size_t i = 0; i < CIFAR_CLASS_NAMES.size(); ++i) {
        std::cout << "  " << i << ": " << CIFAR_CLASS_NAMES[i] << "\n";
    }
    
    std::cout << "\nNote: Full classification requires CIFAR-10 dataset files.\n";
    std::cout << "Download from: https://www.cs.toronto.edu/~kriz/cifar.html\n";
    std::cout << "Place in data/ directory and run with dataset path.\n";
    
    // Demonstrate with synthetic data
    std::cout << "\nSynthetic test:\n";
    CIFARImage test_img;
    test_img.label = 3;  // cat
    for (int row = 0; row < 32; ++row) {
        for (int col = 0; col < 32; ++col) {
            test_img.data[row][col * 3 + 0] = 100;
            test_img.data[row][col * 3 + 1] = 150;
            test_img.data[row][col * 3 + 2] = 200;
        }
    }
    
    PixelArray2D test_array = knn::cifar_to_pixel_array(test_img);
    std::cout << "  Converted CIFAR image to pixel array: " << test_array.size() << "x" << test_array[0].size() << "\n";
}

void demo_performance_comparison() {
    print_separator("Performance Comparison");

    std::cout << "Comparing sequential vs parallel performance\n\n";

    // Use a large array so computation time dominates over thread creation overhead.
    // With small arrays (~10k), thread startup (~1ms each) swamps the actual work,
    // making parallel appear slower. 1M elements gives genuine compute-bound behaviour.
    const size_t size = 1000000;
    PixelArray1D array1(size);
    PixelArray1D array2(size);
    // Fill with non-trivial values so the compiler cannot optimise the loop away
    for (size_t i = 0; i < size; ++i) {
        array1[i] = static_cast<uint8_t>(i % 256);
        array2[i] = static_cast<uint8_t>((i + 37) % 256);
    }

    std::cout << "Array size: " << size << " elements\n\n";

    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    double sequential_time = 0.0;

    // Prepare CSV output
    std::ostringstream csv;
    csv << "threads,time_ms,speedup\n";

    std::cout << "Thread count | Best time (ms) | Speedup\n";
    std::cout << "-------------|----------------|--------\n";

    for (int threads : thread_counts) {
        // Run 5 times, take the best to reduce OS scheduling noise
        constexpr int RUNS = 5;
        double best_ms = std::numeric_limits<double>::max();
        volatile double result = 0.0;  // volatile prevents dead-code elimination
        for (int run = 0; run < RUNS; ++run) {
            auto t0 = std::chrono::high_resolution_clock::now();
            result = algorithm2::euclidean_distance_1d_parallel(array1, array2, threads);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best_ms = std::min(best_ms, ms);
        }
        (void)result;

        if (threads == 1) {
            sequential_time = best_ms;
        }
        double speedup = (best_ms > 0.0) ? sequential_time / best_ms : 1.0;

        std::cout << std::setw(11) << threads << " | "
                  << std::setw(14) << std::fixed << std::setprecision(2) << best_ms << " | "
                  << std::setw(6)  << std::fixed << std::setprecision(2) << speedup << "x\n";

        csv << threads << "," << std::fixed << std::setprecision(4) << best_ms << "," << std::setprecision(4) << speedup << "\n";
    }

    // Write CSV to benchmarks directory
    try {
        std::filesystem::create_directories("benchmarks/results");
        std::ofstream csv_file("benchmarks/results/benchmark_1d.csv");
        if (csv_file.is_open()) {
            csv_file << csv.str();
            csv_file.close();
            std::cout << "\nBenchmark data saved to benchmarks/results/benchmark_1d.csv\n";
        }
    } catch (...) {
        // Silently ignore if we can't write the file (e.g. permissions)
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "CIS6007 - Parallel Image Processing\n";
    std::cout << "Student ID: ST20290339\n";
    std::cout << "========================================\n";
    
    try {
        // Part 1: Algorithm demonstrations
        demo_algorithm1();
        demo_algorithm2();
        demo_algorithm3();
        demo_algorithm4();
        demo_algorithm5();
        demo_parallel_utils();
        
        // Part 2: KNN classifier
        demo_knn_classifier();
        
        // Performance comparison
        demo_performance_comparison();
        
        print_separator("Demonstration Complete");
        std::cout << "All algorithms demonstrated successfully!\n\n";
        std::cout << "To run individual tests:\n";
        std::cout << "  ./test_algorithm1\n";
        std::cout << "  ./test_algorithm2\n";
        std::cout << "  ./test_algorithm3\n";
        std::cout << "  ./test_algorithm4\n";
        std::cout << "  ./test_algorithm5\n";
        std::cout << "  ./test_knn\n\n";
        
        std::cout << "To run KNN classification on CIFAR-10:\n";
        std::cout << "  ./knn_demo <path_to_cifar_data>\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

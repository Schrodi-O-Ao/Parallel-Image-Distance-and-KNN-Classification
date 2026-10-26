#!/bin/bash

# Demo Part 2: KNN Classification on CIFAR-10
# This script demonstrates KNN classification using the CIFAR-10 dataset

echo "========================================"
echo "CIS6007 - Parallel Image Processing"
echo "Part 2: KNN Classification on CIFAR-10"
echo "Student ID: ST20290339"
echo "========================================"
echo ""

# Build if needed
if [ ! -f "build/knn_demo" ] && [ ! -f "build/Release/knn_demo.exe" ]; then
    echo "Executables not found. Building..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
    echo ""
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Data directory not found. Creating it..."
    mkdir -p data
    echo ""
fi

# Check if CIFAR-10 dataset is present (binary version)
DATA_DIR="data/cifar-10-batches-bin"
if [ ! -d "$DATA_DIR" ]; then
    echo "========================================"
    echo "CIFAR-10 Dataset Not Found"
    echo "========================================"
    echo ""
    echo "Please download the CIFAR-10 binary dataset from:"
    echo "https://www.cs.toronto.edu/~kriz/cifar.html"
    echo ""
    echo "Download: cifar-10-binary.tar.gz"
    echo ""
    echo "Extract it to the data/ directory:"
    echo "  tar -xzf cifar-10-binary.tar.gz -C data/"
    echo ""
    echo "Expected structure:"
    echo "  data/cifar-10-batches-bin/"
    echo "    ├── data_batch_1.bin"
    echo "    ├── data_batch_2.bin"
    echo "    ├── data_batch_3.bin"
    echo "    ├── data_batch_4.bin"
    echo "    ├── data_batch_5.bin"
    echo "    ├── test_batch.bin"
    echo "    └── batches.meta.txt"
    echo ""
    echo "After setting up the dataset, run this script again."
    echo ""
    exit 1
fi

echo "CIFAR-10 dataset found at: $DATA_DIR"
echo ""

# Detect executable
if [ -f "build/test_knn" ]; then
    TEST_EXE="./build/test_knn"
    KNN_EXE="./build/knn_demo"
elif [ -f "build/Release/test_knn.exe" ]; then
    TEST_EXE="./build/Release/test_knn.exe"
    KNN_EXE="./build/Release/knn_demo.exe"
else
    echo "ERROR: Could not find test executables."
    exit 1
fi

# Run KNN test
echo "Running KNN Classifier Test"
echo "----------------------------------------"
echo ""
$TEST_EXE
echo ""

echo "========================================"
echo "Full Classification"
echo "========================================"
echo ""
echo "To run full KNN classification on the CIFAR-10 test set:"
echo ""
echo "  $KNN_EXE $DATA_DIR"
echo ""
echo "Note: Full classification may take several minutes"
echo "depending on your system configuration."
echo ""

echo "========================================"
echo "Part 2 Demo Complete"
echo "========================================"
echo ""

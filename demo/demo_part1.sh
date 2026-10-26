#!/bin/bash

# Demo Part 1: Sequential Algorithm Demonstrations
# This script runs demonstrations for all 5 progressive algorithms

echo "========================================"
echo "CIS6007 - Parallel Image Processing"
echo "Part 1: Algorithm Demonstrations"
echo "Student ID: ST20290339"
echo "========================================"
echo ""

# Check if build directory exists and build if needed
if [ ! -f "build/main" ] && [ ! -f "build/Release/main.exe" ]; then
    echo "Main executable not found. Building..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
    echo ""
fi

# Detect executable location (Linux vs Windows/MSVC)
if [ -f "build/main" ]; then
    MAIN_EXE="./build/main"
elif [ -f "build/Release/main.exe" ]; then
    MAIN_EXE="./build/Release/main.exe"
else
    echo "ERROR: Could not find main executable after build."
    exit 1
fi

echo "Running Algorithm Demonstrations"
echo "----------------------------------------"
echo ""

$MAIN_EXE

echo ""
echo "========================================"
echo "Part 1 Demo Complete"
echo "========================================"
echo ""
echo "To run individual algorithm tests:"
echo "  ./build/Release/test_algorithm1  # Algorithm 1: Pixel Comparison"
echo "  ./build/Release/test_algorithm2  # Algorithm 2: 1D Parallel Comparison"
echo "  ./build/Release/test_algorithm3  # Algorithm 3: 2D Parallel Comparison"
echo "  ./build/Release/test_algorithm4  # Algorithm 4: Image Loading"
echo "  ./build/Release/test_algorithm5  # Algorithm 5: Percentage Distance"
echo ""

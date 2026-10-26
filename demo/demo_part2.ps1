# Demo Part 2: KNN Classification on CIFAR-10 (PowerShell)
# Run from the project root directory

Write-Host "========================================"
Write-Host "CIS6007 - Parallel Image Processing"
Write-Host "Part 2: KNN Classification on CIFAR-10"
Write-Host "Student ID: ST20290339"
Write-Host "========================================"
Write-Host ""

# Build if needed
$testExe = "build\Release\test_knn.exe"
$knnExe = "build\Release\knn_demo.exe"
if (-not (Test-Path $testExe)) {
    Write-Host "Executables not found. Building..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
    Write-Host ""
}

# Check dataset
$dataDir = "data\cifar-10-batches-bin"
if (-not (Test-Path $dataDir)) {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "CIFAR-10 Dataset Not Found"
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please download the CIFAR-10 binary dataset from:"
    Write-Host "  https://www.cs.toronto.edu/~kriz/cifar.html"
    Write-Host ""
    Write-Host "Download cifar-10-binary.tar.gz and extract to data/"
    Write-Host ""
    Write-Host "Expected structure:"
    Write-Host "  data\cifar-10-batches-bin\"
    Write-Host "    data_batch_1.bin ... data_batch_5.bin"
    Write-Host "    test_batch.bin"
    Write-Host "    batches.meta.txt"
    exit 1
}

Write-Host "CIFAR-10 dataset found at: $dataDir"
Write-Host ""

if (-not (Test-Path $testExe)) {
    Write-Host "ERROR: Could not find test_knn.exe after build." -ForegroundColor Red
    exit 1
}

Write-Host "Running KNN Classifier Test"
Write-Host "----------------------------------------"
Write-Host ""
& $testExe
Write-Host ""

Write-Host "========================================"
Write-Host "Full Classification"
Write-Host "========================================"
Write-Host ""
Write-Host "To run full KNN classification:"
Write-Host "  .\$knnExe $dataDir"
Write-Host ""

Write-Host "========================================"
Write-Host "Part 2 Demo Complete"
Write-Host "========================================"
Write-Host ""

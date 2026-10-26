# Demo Part 1: Algorithm Demonstrations (PowerShell)
# Run from the project root directory

Write-Host "========================================"
Write-Host "CIS6007 - Parallel Image Processing"
Write-Host "Part 1: Algorithm Demonstrations"
Write-Host "Student ID: ST20290339"
Write-Host "========================================"
Write-Host ""

# Build if needed
$mainExe = "build\Release\main.exe"
if (-not (Test-Path $mainExe)) {
    Write-Host "Main executable not found. Building..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
    Write-Host ""
}

if (-not (Test-Path $mainExe)) {
    Write-Host "ERROR: Could not find main.exe after build." -ForegroundColor Red
    exit 1
}

Write-Host "Running Algorithm Demonstrations"
Write-Host "----------------------------------------"
Write-Host ""

& $mainExe

Write-Host ""
Write-Host "========================================"
Write-Host "Part 1 Demo Complete"
Write-Host "========================================"
Write-Host ""
Write-Host "To run individual algorithm tests:"
Write-Host "  .\build\Release\test_algorithm1.exe"
Write-Host "  .\build\Release\test_algorithm2.exe"
Write-Host "  .\build\Release\test_algorithm3.exe"
Write-Host "  .\build\Release\test_algorithm4.exe"
Write-Host "  .\build\Release\test_algorithm5.exe"
Write-Host ""

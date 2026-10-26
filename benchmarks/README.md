# Benchmarks

This directory contains benchmark tooling for the parallel image processing project.

## Running Benchmarks

1. **Build and run** the main program to generate CSV data:
   ```
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ./build/Release/main.exe
   ```
   This writes timing data to `results/benchmark_1d.csv`.

2. **Generate plots** (requires Python 3 + matplotlib):
   ```
   pip install matplotlib
   python benchmarks/generate_plots.py
   ```
   If no CSV exists, the script uses embedded sample data.

## Output

- `results/benchmark_1d.csv`: raw timing data
- `results/speedup_curve.png`: measured speedup vs Amdahl's Law prediction
- `results/execution_time.png`: wall-clock time per thread count
- `results/efficiency.png`: parallel efficiency (speedup / threads)

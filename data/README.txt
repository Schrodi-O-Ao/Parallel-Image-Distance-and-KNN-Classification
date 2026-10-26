# Dataset Setup Instructions

## CIFAR-10 Dataset

This project uses the CIFAR-10 dataset for KNN classification. Follow these instructions to set up the dataset.

### Dataset Information

- **Name:** CIFAR-10
- **Size:** 60,000 images (50,000 training, 10,000 test)
- **Image Dimensions:** 32x32 pixels, 3 channels (RGB)
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Format:** Binary files

### Download Instructions

1. Visit the CIFAR-10 dataset website:
   https://www.cs.toronto.edu/~kriz/cifar.html

2. Download the binary version:
   - File: `cifar-10-binary.tar.gz`
   - Size: ~162 MB (compressed)
   - URL: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

### Installation

#### Option 1: Using wget (Linux/macOS)
```bash
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
rm cifar-10-binary.tar.gz
```

#### Option 2: Using curl (Linux/macOS)
```bash
cd data
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
rm cifar-10-binary.tar.gz
```

#### Option 3: Manual Download
1. Download `cifar-10-binary.tar.gz` from the website
2. Extract the archive to the `data/` directory
3. Ensure the extracted folder is named `cifar-10-batches-bin`

### Expected Directory Structure

After extraction, your `data/` directory should look like this:

```
data/
├── README.txt
└── cifar-10-batches-bin/
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    ├── data_batch_5.bin
    ├── test_batch.bin
    ├── batches.meta.txt
    └── readme.html
```

### File Descriptions

- **data_batch_1.bin** through **data_batch_5.bin**: Training data batches (10,000 images each)
- **test_batch.bin**: Test data (10,000 images)
- **batches.meta.txt**: Metadata file containing class names

### Binary File Format

Each batch file contains 10,000 images in the following format:

```
<1 byte label><3072 bytes pixel data>
```

- **Label (1 byte):** Integer 0-9 representing the class
- **Pixel data (3072 bytes):**
  - First 1024 bytes: Red channel values
  - Next 1024 bytes: Green channel values
  - Last 1024 bytes: Blue channel values
  - Values are stored in row-major order

### Class Labels

The 10 classes in CIFAR-10 are:

| Label | Class Name    |
|-------|---------------|
| 0     | airplane      |
| 1     | automobile    |
| 2     | bird          |
| 3     | cat           |
| 4     | deer          |
| 5     | dog           |
| 6     | frog          |
| 7     | horse         |
| 8     | ship          |
| 9     | truck         |

### Running KNN Classification

Once the dataset is set up, you can run the KNN classifier:

```bash
# Using the demo script
./demo/demo_part2.sh

# Or directly with the main program
./build/main data/cifar-10-batches-bin
```

### Performance Considerations

- **Full classification** of all 10,000 test images may take:
  - ~30-60 minutes (sequential)
  - ~8-15 minutes (4 threads)
  - ~4-8 minutes (8 threads)
  - ~2-4 minutes (16 threads)

- **Single image classification** takes ~0.5-2 seconds depending on thread count

- **Memory usage:** ~500 MB for loading the full training set

### Troubleshooting

#### Dataset not found
If you see "Dataset not found" errors:
1. Verify the dataset is in `data/cifar-10-batches-bin/`
2. Check that all 6 binary files are present
3. Ensure file permissions allow reading

#### Build errors
If you encounter build errors:
1. Ensure OpenCV is installed correctly
2. Check that CMake or Make is properly configured
3. Verify C++20 compiler support

#### Runtime errors
If the program crashes during classification:
1. Check available memory (need at least 1 GB free)
2. Verify dataset integrity (re-download if corrupted)
3. Try with fewer threads if using many cores

### Citation

If you use this dataset in your research, please cite:

```
@techreport{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  year={2009},
  institution={University of Toronto}
}
```

### License

The CIFAR-10 dataset is freely available for research purposes. See the official website for more details.

### Additional Resources

- Official CIFAR-10 website: https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10 documentation: https://www.cs.toronto.edu/~kriz/cifar.html
- Dataset paper: https://www.cs.toronto.edu/~techrep/2009/techrep_hinton_09.pdf

### Sample Images

For development and testing without downloading the full dataset, you can place sample test images in the `data/sample/` directory. These should be in standard image formats (PNG, JPG, BMP) for use with Algorithm 4.

Note: Sample images are only for testing Algorithm 4 (image loading). KNN classification requires the full CIFAR-10 binary dataset.

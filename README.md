# Image Compression Analysis using Sequential and Parallel Computing

A C++-based benchmarking project for studying the computational cost of JPEG compression and reconstruction on increasingly large image collections, with a focus on comparing a sequential implementation against an OpenMP-parallel implementation.

This repository evaluates how image processing cost, reconstruction quality, and scalability change as the dataset size increases. The goal is not only to compress images, but to measure the trade-off between image quality, storage reduction, and execution time under different workloads.

---

## Overview

Image compression is widely used to reduce storage requirements and improve data transfer efficiency. In large image collections, processing each image sequentially can become computationally expensive. This project investigates that problem by compressing and reconstructing a dataset of natural images and comparing performance between:

1. A sequential implementation that processes images one after another.
2. A parallel implementation using OpenMP that processes independent images concurrently.

The same dataset, selection strategy, JPEG quality setting, and evaluation metrics are used across both implementations so the comparison remains fair and reproducible.

### Primary objectives

- Compress a large collection of JPEG images using TurboJPEG.
- Decompress the compressed images and reconstruct the original content.
- Measure compression quality using MSE, PSNR, and SSIM.
- Measure storage savings and compression ratio.
- Measure processing time for compression and decompression.
- Compare sequential and parallel performance using execution time, speedup, and efficiency.
- Evaluate how performance scales as the number of images increases.
- Identify bottlenecks and practical limitations of parallel image processing.

---

## Problem Statement

Image collections can be large, and compressing each image individually imposes a significant computational cost. This project evaluates how JPEG re-compression behaves when applied across many images and determines whether OpenMP parallelization improves execution time without sacrificing output quality.

The study analyzes the relationship between:

- dataset size,
- compression quality,
- file size reduction,
- reconstruction quality,
- runtime performance.

---

## Research Methodology

The project follows a consistent processing pipeline:

```text
Dataset
  ↓
Select image subset
  ↓
Read image
  ↓
JPEG compression
  ↓
Save compressed JPEG bytes
  ↓
JPEG decompression
  ↓
Reconstruct image
  ↓
Compute quality metrics
  ↓
Compute storage metrics
  ↓
Record time measurements
  ↓
Write results to CSV
```

### Sequential pipeline

```text
Image 1 → Compress → Decompress
Image 2 → Compress → Decompress
Image 3 → Compress → Decompress
...
Image N → Compress → Decompress
```

### Parallel pipeline

```text
Threads → Image 1 → Compress → Decompress
        → Image 2 → Compress → Decompress
        → Image 3 → Compress → Decompress
        → Image 4 → Compress → Decompress
        → ...
```

Because each image is processed independently, image-level parallelism is the natural OpenMP strategy.

---

## Dataset

The project uses the Natural Images dataset, which contains images organized into eight classes:

```text
airplane
car
cat
dog
flower
fruit
motorbike
person
```

Each folder contains 500 JPEG images, giving a total of:

```text
8 classes × 500 images = 4000 images
```

Example filenames are:

```text
airplane_0000.jpg
airplane_0001.jpg
airplane_0002.jpg
...
```

### Deterministic image selection

The dataset is not split into separate physical copies for each experiment. Instead, the programs select the first `N` sorted images from each class. This creates nested, repeatable, and fair comparisons.

For example:

```text
50 images/class
      ↓
100 images/class
      ↓
250 images/class
      ↓
500 images/class
```

The same ordering and selection rule is used in both the sequential and parallel implementations.

---

## Experiments

The main experiments are based on the number of images processed per class.

| Experiment | Images/Class | Classes | Total Images |
|------------|-------------:|--------:|-------------:|
| E1         | 50           | 8       | 400          |
| E2         | 100          | 8       | 800          |
| E3         | 250          | 8       | 2000         |
| E4         | 500          | 8       | 4000         |

The JPEG quality is kept fixed throughout the main scalability experiment:

```text
JPEG Quality = 75
```

The project also supports a small correctness test with exactly 2 total images.

---

## Correctness Test

A small validation run is available to confirm the pipeline behaves correctly before larger experiments are launched.

```bash
./sequential.exe 2
```

This test processes exactly two images and verifies that:

- image files can be discovered and read correctly,
- JPEG compression succeeds,
- JPEG decompression succeeds,
- reconstructed images are written to disk,
- MSE is computed,
- PSNR is computed,
- SSIM is computed,
- CSV output is written successfully.

This should be run before starting large-scale benchmarking.

---

## Compression Method

The project uses JPEG compression via TurboJPEG / libjpeg-turbo.

The operating pipeline is:

```text
Original JPEG image
        ↓
Load with OpenCV
        ↓
Convert BGR → RGB
        ↓
TurboJPEG encode
        ↓
Compressed JPEG bytes
        ↓
TurboJPEG decode
        ↓
Reconstructed image
```

### Important note

The dataset already contains JPEG images. Therefore, this project is studying JPEG re-compression of JPEG inputs rather than comparing an uncompressed image to its first JPEG encoding. This is important when interpreting compression ratio and quality results.

---

## Evaluation Metrics

The software records both storage-related metrics and image-quality metrics.

### 1. Compression ratio

```text
Compression Ratio = Original Size / Compressed Size
```

A higher value indicates more aggressive reduction relative to the original file size.

### 2. Space saving

```text
Space Saving (%) = ((Original Size - Compressed Size) / Original Size) × 100
```

This measures the percentage of storage saved after compression.

### 3. Mean Squared Error (MSE)

```text
MSE = (1 / N) × Σ (original - reconstructed)^2
```

MSE computes the average squared pixel difference between the original and reconstructed image. Lower MSE indicates better reconstruction fidelity.

### 4. Peak Signal-to-Noise Ratio (PSNR)

```text
PSNR = 10 × log10((MAX^2) / MSE)
```

where `MAX = 255` for 8-bit images. Higher PSNR generally indicates lower reconstruction error.

### 5. Structural Similarity Index (SSIM)

SSIM compares local structure, luminance, and contrast between the original and reconstructed images. It is often more descriptive than MSE or PSNR for visual image quality.

In this project, SSIM is calculated using a global grayscale formulation that is kept consistent between the sequential and parallel implementations.

---

## Performance Metrics

The project records timings for:

- compression time,
- decompression time,
- total processing time.

The timing is measured around the TurboJPEG encode/decode operations, with the same measurement strategy used in the sequential and parallel versions.

### Speedup

```text
Speedup = Sequential Time / Parallel Time
```

### Parallel efficiency

```text
Parallel Efficiency (%) = (Speedup / Number of Threads) × 100
```

### Thread scaling

The OpenMP version is tested using multiple thread counts such as:

```text
1 thread
2 threads
4 threads
8 threads
```

The actual thread count used for a given run is recorded in the CSV output and benchmark file.

---

## Repository Structure

```text
Image_Compression_Analysis-main/
├── README.md
├── verify.py
├── natural_images/
│   ├── airplane/
│   ├── car/
│   ├── cat/
│   ├── dog/
│   ├── flower/
│   ├── fruit/
│   ├── motorbike/
│   └── person/
├── src/
│   ├── sequential.cpp
│   └── parallel.cpp
├── output/
│   ├── compressed/
│   │   ├── 2/
│   │   ├── 50/
│   │   ├── 100/
│   │   ├── 250/
│   │   └── 500/
│   ├── reconstructed/
│   │   ├── 2/
│   │   ├── 50/
│   │   ├── 100/
│   │   ├── 250/
│   │   └── 500/
│   ├── parallel_compressed/
│   │   ├── 50/
│   │   ├── 100/
│   │   ├── 250/
│   │   └── 500/
│   └── parallel_reconstructed/
│       ├── 50/
│       ├── 100/
│       ├── 250/
│       └── 500/
├── results/
│   ├── sequential_2_per_class.csv
│   ├── sequential_50_per_class.csv
│   ├── sequential_100_per_class.csv
│   ├── sequential_250_per_class.csv
│   ├── sequential_500_per_class.csv
│   ├── parallel_50_t1_per_class.csv
│   ├── parallel_50_t2_per_class.csv
│   ├── parallel_50_t4_per_class.csv
│   ├── parallel_50_t8_per_class.csv
│   └── parallel_benchmark.csv
└── test/
    └── test.cpp
```

---

## Requirements

### Hardware

- Multi-core CPU recommended
- At least 8 GB RAM recommended
- Sufficient disk space for the dataset and generated outputs

### Software

The project is developed under:

- Windows
- MSYS2 UCRT64 shell
- GCC / G++
- C++17
- OpenCV
- TurboJPEG / libjpeg-turbo
- OpenMP
- Visual Studio Code

---

## Dependencies

### C++ standard library

The code relies on:

```cpp
<iostream>
<fstream>
<vector>
<string>
<filesystem>
<algorithm>
<chrono>
<cmath>
<iomanip>
```

### OpenCV

Used for:

- image loading,
- color-space conversion,
- matrix operations,
- image writing,
- basic image processing utilities.

### TurboJPEG / libjpeg-turbo

Used for:

- JPEG compression,
- JPEG decompression,
- efficient compressed byte generation.

### OpenMP

Used by the parallel implementation for shared-memory parallelism across CPU threads.

---

## Setup and Installation

The following instructions are tailored for an MSYS2 UCRT64 environment on Windows.

### 1. Install GCC

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```

Verify:

```bash
g++ --version
```

### 2. Install OpenCV

```bash
pacman -S mingw-w64-ucrt-x86_64-opencv
```

### 3. Install TurboJPEG

```bash
pacman -S mingw-w64-ucrt-x86_64-libjpeg-turbo
```

### 4. Confirm compiler support

OpenMP support is available through GCC. The compiler is invoked with:

```text
-fopenmp
```

---

## Building the Project

From the repository root, compile both implementations:

```bash
g++ -std=c++17 -O3 -fopenmp src/sequential.cpp -o sequential.exe $(pkg-config --cflags --libs opencv4) -lturbojpeg
g++ -std=c++17 -O3 -fopenmp src/parallel.cpp -o parallel.exe $(pkg-config --cflags --libs opencv4) -lturbojpeg
```

If `pkg-config` is not available in your setup, you can also compile using explicit include and library paths as needed for your environment.

---

## Running the Experiments

### Sequential version

Run all standard experiments:

```bash
./sequential.exe
```

Run only the correctness test:

```bash
./sequential.exe 2
```

Run a single experiment size:

```bash
./sequential.exe 50
./sequential.exe 100
./sequential.exe 250
./sequential.exe 500
```

### Parallel version

Usage:

```bash
./parallel.exe <images_per_class> <threads>
```

Examples:

```bash
./parallel.exe 50 1
./parallel.exe 50 2
./parallel.exe 50 4
./parallel.exe 50 8
```

Accepted image counts:

```text
2, 50, 100, 250, 500
```

Accepted thread counts:

```text
1, 2, 4, 8
```

---

## Output Files

### Sequential output

The sequential implementation stores output in:

```text
output/reconstructed/<images_per_class>/<class_name>/
```

and writes per-image CSV metrics to:

```text
results/sequential_<images_per_class>_per_class.csv
```

Example:

```text
results/sequential_50_per_class.csv
```

### Parallel output

The parallel version stores reconstructed images in:

```text
output/parallel_reconstructed/<images_per_class>/t<threads>/<class_name>/
```

and writes CSV results to:

```text
results/parallel_<images_per_class>_t<threads>_per_class.csv
```

Example:

```text
results/parallel_100_t4_per_class.csv
```

A benchmark summary is also appended to:

```text
results/parallel_benchmark.csv
```

---

## Correctness Verification

A Python verification script is included to compare sequential and parallel outputs for the same experiment.

Usage:

```bash
python verify.py 50 4
```

This validates:

- same image set,
- same metric values,
- same reconstructed image pixels,
- same dataset and result counts.

The script compares the CSV data and reconstructed images in `output` and reports whether the parallel result matches the sequential one.

---

## Performance Interpretation

The project compares execution time and output quality under consistent conditions. A lower time is better, but it must also be considered alongside:

- image quality metrics,
- compression ratio,
- output fidelity,
- dataset size,
- hardware/thread count.

In practice, the expected outcome is that parallel execution reduces wall-clock time as the image count increases, while maintaining essentially identical quality metrics to the sequential version.

---

## Notes and Assumptions

- The project uses JPEG input images; therefore, it measures JPEG re-compression, not raw-image-to-JPEG conversion.
- The same deterministic sorting and selection strategy is used across all runs.
- Output quality is computed after re-encoding and decoding, which reflects real pipeline behavior.
- The benchmark includes image encode/decode time but not unrelated system-level overhead outside the measured algorithmic workflow.

---

## Summary

This repository provides a complete benchmark pipeline for comparing sequential and OpenMP-parallel JPEG image compression and reconstruction. It is designed to be reproducible, deterministic, and suitable for analyzing the performance of large image collections under consistent experimental conditions.

The software records both numerical quality metrics and runtime performance, making it useful for evaluating the effectiveness of parallel processing on image workloads.

---

## Quick Start

```bash
# Build
 g++ -std=c++17 -O3 -fopenmp src/sequential.cpp -o sequential.exe $(pkg-config --cflags --libs opencv4) -lturbojpeg
 g++ -std=c++17 -O3 -fopenmp src/parallel.cpp -o parallel.exe $(pkg-config --cflags --libs opencv4) -lturbojpeg

# Correctness test
 ./sequential.exe 2

# Sequential full experiment
 ./sequential.exe 50

# Parallel benchmark example
 ./parallel.exe 50 4

# Verify outputs
 python verify.py 50 4
```

---

## License

This project is provided for academic and research use as part of a computational image processing and parallel computing study.

---




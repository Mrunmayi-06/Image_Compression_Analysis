# Image Compression Analysis using Sequential and Parallel Computing

A C++-based image compression project that studies the computational cost of JPEG compression and reconstruction on increasing image collections, with a focus on **sequential versus OpenMP parallel execution**.

---

## Overview

Image compression is widely used to reduce storage requirements and data-transfer costs. However, processing a large collection of images can require significant computational time when every image is processed sequentially.

This project implements an image compression and reconstruction pipeline and evaluates how its performance changes as the number of images increases.

The project has two main implementations:

1. **Sequential implementation** — processes images one after another.
2. **Parallel implementation using OpenMP** — distributes independent image-processing tasks across multiple CPU threads.

The implementations are evaluated using the same dataset, compression settings, image-selection strategy, and quality metrics so that their performance can be compared fairly.

### Main goals

* Compress a large collection of images using JPEG.
* Reconstruct/decompress the compressed images.
* Measure compression and reconstruction quality.
* Measure execution time for different dataset sizes.
* Parallelize the workload using OpenMP.
* Compare sequential and parallel execution.
* Measure speedup and parallel efficiency.
* Study scalability as the dataset size increases.
* Identify computational bottlenecks and limitations.

---

# Problem Statement

Develop and evaluate a computational approach for compressing large collections of images while maintaining acceptable reconstruction quality. Investigate the computational requirements for increasing image collections and determine how parallel processing can affect execution time and scalability.

---

# Approach

The project follows the pipeline:

```text
                    IMAGE DATASET
                         |
                         v
                  Image Selection
                         |
                         v
                  Read Input Image
                         |
                         v
                  JPEG Compression
                         |
                         v
                  Compressed Image
                         |
                         v
                 JPEG Decompression
                         |
                         v
                Reconstructed Image
                         |
                         v
              Quality + Size Metrics
                         |
                         v
                 Performance Results
```

The sequential implementation performs the image operations one after another.

The OpenMP implementation parallelizes independent image-processing operations.

```text
Sequential

Image 1 ──> Compress ──> Decompress
Image 2 ──> Compress ──> Decompress
Image 3 ──> Compress ──> Decompress
...
Image N ──> Compress ──> Decompress
```

```text
OpenMP

             ┌──> Image 1 ──> Compress ──> Decompress
             ├──> Image 2 ──> Compress ──> Decompress
CPU Threads ─┼──> Image 3 ──> Compress ──> Decompress
             ├──> Image 4 ──> Compress ──> Decompress
             └──> ...
```

Because individual images can be processed independently, image-level parallelism is used as the primary parallelization strategy.

---

# Dataset

The project uses the **Natural Images** dataset.

The dataset contains eight image categories:

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

Each category contains 500 images.

Therefore:

```text
8 classes × 500 images = 4000 images
```

Example filenames:

```text
airplane_0000.jpg
airplane_0001.jpg
airplane_0002.jpg
...
```

The original dataset is kept unchanged.

## Dataset Selection

The project does not create separate physical datasets for every experiment.

Instead, the program:

1. Reads the images from each class directory.
2. Sorts the filenames alphabetically.
3. Selects the first `N` images required for the experiment.

This produces nested and reproducible experiments.

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

The same selection rule will be used by the sequential and parallel implementations.

---

# Experiments

The main experiments are:

| Experiment | Images / Class | Number of Classes | Total Images |
| ---------- | -------------: | ----------------: | -----------: |
| E1         |             50 |                 8 |          400 |
| E2         |            100 |                 8 |          800 |
| E3         |            250 |                 8 |         2000 |
| E4         |            500 |                 8 |         4000 |

The primary JPEG quality setting is:

```text
JPEG Quality = 75
```

The main scalability experiment keeps the JPEG quality fixed while increasing the number of images.

Additional quality experiments may be added later by the team.

---

# Correctness Test

Before running the large experiments, the sequential program supports a small correctness test.

```bash
./sequential.exe 2
```

This processes exactly two images.

The test verifies that:

* Images can be located correctly.
* Images can be read successfully.
* JPEG compression works.
* Compressed data can be decompressed.
* Reconstructed images can be generated.
* MSE can be calculated.
* PSNR can be calculated.
* SSIM can be calculated.
* Output files are generated.
* CSV results are written successfully.

A successful correctness test should be performed before starting the full experiments.

---

# Compression

The current implementation uses JPEG compression with:

```text
Quality = 75
```

TurboJPEG/libjpeg-turbo is used for the JPEG encoding and decoding operations.

The current compression pipeline is:

```text
Original JPEG
     |
     v
Decode / Load Image
     |
     v
JPEG Re-encoding
     |
     v
Compressed JPEG Data
     |
     v
JPEG Decoding
     |
     v
Reconstructed Image
```

## Important Dataset Note

The Natural Images dataset contains JPEG images.

Therefore, this project performs **JPEG re-compression/re-encoding of JPEG input images**.

It is not a comparison between an uncompressed raw image and its first JPEG encoding.

This distinction should be considered when interpreting the compression results.

---

# Evaluation Metrics

The project evaluates both storage reduction and reconstructed image quality.

## Compression Ratio

```text
Compression Ratio =
Original Size / Compressed Size
```

## Space Saving

```text
Space Saving (%) =
((Original Size - Compressed Size) / Original Size) × 100
```

## Mean Squared Error — MSE

MSE measures the average squared difference between the original and reconstructed images.

Lower MSE indicates lower pixel-level reconstruction error.

## Peak Signal-to-Noise Ratio — PSNR

PSNR measures reconstruction quality based on pixel error.

Higher PSNR generally indicates lower reconstruction error.

## Structural Similarity Index — SSIM

SSIM measures structural similarity between the original and reconstructed images.

The current sequential implementation uses a custom global grayscale SSIM calculation.

The exact SSIM implementation should remain consistent between sequential and parallel versions.

---

# Performance Metrics

The following timing measurements are collected:

* Compression time
* Decompression time
* Total processing time

The current timing scope measures the TurboJPEG compression and decompression operations.

File scanning, output writing and other setup operations should be kept consistent when comparing sequential and parallel implementations.

---

# Parallel Performance

The parallel implementation uses OpenMP.

The main performance measurements will include:

### Execution Time

```text
Sequential execution time
Parallel execution time
```

### Speedup

```text
Speedup =
Sequential Time / Parallel Time
```

### Parallel Efficiency

```text
Parallel Efficiency (%) =
Speedup / Number of Threads × 100
```

### Thread Scaling

The parallel implementation will be tested with multiple thread counts, depending on the available CPU hardware.

Example:

```text
1 thread
2 threads
4 threads
8 threads
```

The actual thread counts used will be recorded in the final experiment results.

---

# Project Structure

The project is organized so that individual team members can add their components without changing the existing dataset.

```text
ImageCompressionProject/
│
├── natural_images/
│   ├── airplane/
│   ├── car/
│   ├── cat/
│   ├── dog/
│   ├── flower/
│   ├── fruit/
│   ├── motorbike/
│   └── person/
│
├── src/
│   ├── sequential.cpp
│   ├── sequential.exe
│   └── parallel.cpp
│
├── output/
│   ├── compressed/
│   │   ├── 2/
│   │   ├── 50/
│   │   ├── 100/
│   │   ├── 250/
│   │   └── 500/
│   │
│   └── reconstructed/
│       ├── 2/
│       ├── 50/
│       ├── 100/
│       ├── 250/
│       └── 500/
│
├── results/
│   ├── sequential_2_per_class.csv
│   ├── sequential_50_per_class.csv
│   ├── sequential_100_per_class.csv
│   ├── sequential_250_per_class.csv
│   └── sequential_500_per_class.csv
│
├── README.md
└── ...
```

As the project progresses, additional files can be added for:

```text
parallel implementation
performance analysis
graphs
experiment scripts
documentation
LLM usage log
```

---

# Requirements

## Hardware

The project is CPU-based.

Recommended:

* Multi-core CPU
* At least 8 GB RAM
* Sufficient disk space for the dataset and generated outputs

The exact CPU model and RAM used for the final experiments should be recorded in the experimental setup.

## Software

Current development environment:

* Windows
* MSYS2 UCRT64
* GCC/G++ 16.2.0
* C++17
* OpenCV 5.0.0
* libjpeg-turbo / TurboJPEG
* OpenMP
* Visual Studio Code

---

# Dependencies

The project currently uses:

### C++ Standard Library

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

* Image representation
* Image conversion
* Image processing
* Matrix operations

### TurboJPEG / libjpeg-turbo

Used for:

* JPEG compression
* JPEG decompression

### OpenMP

Used by the parallel implementation for shared-memory CPU parallelism.

---

# Installing the Dependencies

The following commands are for an MSYS2 UCRT64 environment.

## GCC

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```

Verify:

```bash
g++ --version
```

## OpenCV

```bash
pacman -S mingw-w64-ucrt-x86_64-opencv
```

## libjpeg-turbo / TurboJPEG

```bash
pacman -S mingw-w64-ucrt-x86_64-libjpeg-turbo
```

## OpenMP

OpenMP support is provided through GCC.

The compiler should be invoked with:

```text
-fopenmp
```

---

# Building the Sequential Implementation

Open the MSYS2 UCRT64 terminal and move to the `src` directory.

```bash
cd /c/Users/<username>/OneDrive/Desktop/daat1/src
```

Compile:

```bash
g++ sequential.cpp -o sequential.exe -std=c++17 \
-I/ucrt64/include/opencv5 \
-L/ucrt64/lib \
-lopencv_core \
-lopencv_imgcodecs \
-lopencv_imgproc \
-lopencv_quality \
-lturbojpeg \
-ljpeg \
-fopenmp
```

Run the correctness test:

```bash
./sequential.exe 2
```

---

# Running Experiments

## Correctness Test

```bash
./sequential.exe 2
```

Processes exactly two images.

---

## Experiment E1

```bash
./sequential.exe 50
```

Processes:

```text
50 × 8 = 400 images
```

---

## Experiment E2

```bash
./sequential.exe 100
```

Processes:

```text
100 × 8 = 800 images
```

---

## Experiment E3

```bash
./sequential.exe 250
```

Processes:

```text
250 × 8 = 2000 images
```

---

## Experiment E4

```bash
./sequential.exe 500
```

Processes:

```text
500 × 8 = 4000 images
```

---

## Run All Sequential Experiments

Running the executable without an argument runs the complete experiment set:

```bash
./sequential.exe
```

This runs:

```text
E1 → 50 images/class
E2 → 100 images/class
E3 → 250 images/class
E4 → 500 images/class
```

---

# Output

The program creates compressed and reconstructed images under:

```text
output/compressed/
output/reconstructed/
```

For example:

```text
output/
├── compressed/
│   └── 50/
│       ├── airplane/
│       ├── car/
│       ├── cat/
│       └── ...
│
└── reconstructed/
    └── 50/
        ├── airplane/
        ├── car/
        ├── cat/
        └── ...
```

The program also creates CSV files under:

```text
results/
```

Example:

```text
sequential_50_per_class.csv
```

---

# CSV Results

The current sequential implementation records:

```text
Class
Image
Original_Size_Bytes
Compressed_Size_Bytes
Compression_Ratio
Space_Saving_Percent
Compression_Time_ms
Decompression_Time_ms
Total_Time_ms
MSE
PSNR_dB
SSIM
```

The CSV contains both per-image measurements and aggregate experiment results.

The CSV files will be used later to generate:

* Performance tables
* Execution-time graphs
* Compression-ratio graphs
* Quality graphs
* Dataset-size scaling graphs
* Sequential-versus-parallel comparisons

---

# Experimental Methodology

For a fair comparison, sequential and parallel implementations should use:

* The same dataset
* The same selected images
* The same JPEG quality
* The same compression algorithm
* The same reconstruction process
* The same quality metrics
* The same machine
* The same input/output conditions
* Consistent timing methodology

Each major experiment should ideally be repeated multiple times.

The final report should record the average or otherwise clearly defined representative execution time.

---

# Correctness Verification

Correctness will be checked at multiple levels.

### Functional correctness

Verify that:

* Every selected input image is processed.
* Every image produces a compressed output.
* Every compressed image can be reconstructed.
* CSV entries are generated correctly.

### Quality correctness

Verify that:

* MSE is finite and non-negative.
* PSNR is calculated correctly.
* SSIM remains within its expected range.
* Original and reconstructed images have compatible dimensions for comparison.

### Sequential vs Parallel correctness

The parallel implementation should process the same inputs and produce equivalent compression and reconstruction results under the same settings.

Small differences caused by implementation/library behavior should be investigated and documented rather than ignored.

---

# Performance Analysis Plan

The final analysis will investigate three major dimensions.

## 1. Dataset Size Scaling

Compare:

```text
400 images
800 images
2000 images
4000 images
```

Measure how execution time changes as the workload increases.

## 2. Thread Scaling

For a fixed dataset size, compare multiple OpenMP thread counts.

Example:

```text
1
2
4
8
```

## 3. Compression Quality

If included in the final experiments, compare different JPEG quality levels while observing:

* File size
* Compression ratio
* Space saving
* MSE
* PSNR
* SSIM
* Execution time

The primary scalability experiment uses JPEG quality 75.

---

# Bottleneck Analysis

The final project will investigate where execution time is spent.

Potential areas include:

* JPEG encoding
* JPEG decoding
* Image loading
* Memory allocation
* Memory bandwidth
* Disk I/O
* Thread scheduling
* Synchronization
* Output file writing

The actual bottlenecks will be identified from measured results rather than assumed beforehand.

---

# Current Status

## Sequential Implementation

### Completed

* Dataset organization
* Dataset selection strategy
* C++ environment setup
* OpenCV installation
* TurboJPEG installation
* JPEG compression
* JPEG decompression
* Reconstruction
* Compression ratio calculation
* Space saving calculation
* MSE calculation
* PSNR calculation
* SSIM calculation
* Execution-time measurement
* CSV generation
* Correctness test
* Experiment modes for 50, 100, 250 and 500 images/class

### Verified

The small correctness test has successfully processed two images and generated:

* Compressed output
* Reconstructed output
* Compression statistics
* Quality metrics
* CSV results

---

# Parallel Implementation

The parallel implementation is planned using OpenMP.

It will:

1. Reuse the sequential processing logic where appropriate.
2. Identify independent image-level operations.
3. Parallelize those operations using OpenMP.
4. Support configurable thread counts.
5. Generate results using a format compatible with the sequential results.
6. Verify correctness against the sequential implementation.
7. Record parallel execution time.
8. Calculate speedup and efficiency.

This section should be updated by the team member responsible for the parallel implementation.

---
# License

This project is developed as an academic project.

The Natural Images dataset is not included in this repository. Users should obtain the dataset separately and comply with its applicable terms of use.

---

# Acknowledgements

This project uses open-source software libraries including:

* OpenCV
* libjpeg-turbo / TurboJPEG
* OpenMP
* GCC

The project team acknowledges the developers and maintainers of these tools and libraries.



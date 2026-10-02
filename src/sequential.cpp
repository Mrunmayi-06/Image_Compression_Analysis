#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <turbojpeg.h>

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// CONSTANTS
// ============================================================

// JPEG compression quality
const int JPEG_QUALITY = 75;

// Number of images to select from each class
// Full experiments use 50, 100, 250 and 500 images per class.
// A command-line value of 2 runs a small correctness test using exactly
// 2 images total.
const vector<int> IMAGE_COUNTS = {50, 100, 250, 500};

// The 8 class folders in the Natural Images dataset
const vector<string> CLASS_NAMES =
{
    "airplane",
    "car",
    "cat",
    "dog",
    "flower",
    "fruit",
    "motorbike",
    "person"
};

// ============================================================
// STRUCTURE TO STORE RESULTS FOR ONE IMAGE
// ============================================================

struct ImageResult
{
    string className;
    string imageName;

    long long originalSize;
    long long compressedSize;

    double compressionRatio;
    double spaceSaving;

    double compressionTime;
    double decompressionTime;
    double totalTime;

    double mse;
    double psnr;
    double ssim;
};


// ============================================================
// FUNCTION: Calculate MSE
// ============================================================

double calculateMSE(const cv::Mat& original, const cv::Mat& reconstructed)
{
    // Make sure both images have the same size
    if (original.size() != reconstructed.size())
    {
        return -1.0;
    }

    // Convert images to floating point
    cv::Mat originalFloat;
    cv::Mat reconstructedFloat;

    original.convertTo(originalFloat, CV_32F);
    reconstructed.convertTo(reconstructedFloat, CV_32F);

    // Difference between original and reconstructed image
    cv::Mat difference = originalFloat - reconstructedFloat;

    // Square the difference
    cv::Mat squaredDifference;
    cv::multiply(difference, difference, squaredDifference);

    // Calculate mean squared error
    cv::Scalar sum = cv::sum(squaredDifference);

    double totalPixels =
        static_cast<double>(
            original.total() * original.channels()
        );

    double mse = (sum[0] + sum[1] + sum[2]) / totalPixels;

    return mse;
}


// ============================================================
// FUNCTION: Calculate PSNR
// ============================================================

double calculatePSNR(double mse)
{
    // If MSE is zero, images are identical
    if (mse == 0.0)
    {
        return INFINITY;
    }

    // Maximum pixel value for an 8-bit image
    const double MAX_PIXEL = 255.0;

    double psnr =
        10.0 * log10(
            (MAX_PIXEL * MAX_PIXEL) / mse
        );

    return psnr;
}


// ============================================================
// FUNCTION: Calculate SSIM
//
// This is a simple global SSIM calculation.
// It calculates SSIM using luminance, contrast and structure.
// ============================================================

double calculateSSIM(const cv::Mat& original, const cv::Mat& reconstructed)
{
    if (original.empty() || reconstructed.empty())
    {
        return -1.0;
    }

    if (original.size() != reconstructed.size())
    {
        return -1.0;
    }

    // Convert BGR images to grayscale
    cv::Mat originalGray;
    cv::Mat reconstructedGray;

    cv::cvtColor(original, originalGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(reconstructed, reconstructedGray, cv::COLOR_BGR2GRAY);

    // Convert to double for calculations
    cv::Mat originalDouble;
    cv::Mat reconstructedDouble;

    originalGray.convertTo(originalDouble, CV_64F);
    reconstructedGray.convertTo(reconstructedDouble, CV_64F);

    // Calculate means
    cv::Scalar meanOriginal;
    cv::Scalar meanReconstructed;

    cv::meanStdDev(
        originalDouble,
        meanOriginal,
        cv::noArray()
    );

    cv::meanStdDev(
        reconstructedDouble,
        meanReconstructed,
        cv::noArray()
    );

    double muX = meanOriginal[0];
    double muY = meanReconstructed[0];

    // Calculate variance and covariance

    cv::Mat diffOriginal = originalDouble - muX;
    cv::Mat diffReconstructed = reconstructedDouble - muY;

    double varianceOriginal =
        cv::sum(diffOriginal.mul(diffOriginal))[0]
        / (originalDouble.total() - 1);

    double varianceReconstructed =
        cv::sum(diffReconstructed.mul(diffReconstructed))[0]
        / (reconstructedDouble.total() - 1);

    double covariance =
        cv::sum(
            diffOriginal.mul(diffReconstructed)
        )[0]
        / (originalDouble.total() - 1);

    // SSIM constants
    const double L = 255.0;

    const double C1 =
        (0.01 * L) * (0.01 * L);

    const double C2 =
        (0.03 * L) * (0.03 * L);

    // SSIM formula
    double numerator =
        (2.0 * muX * muY + C1)
        *
        (2.0 * covariance + C2);

    double denominator =
        (muX * muX + muY * muY + C1)
        *
        (varianceOriginal + varianceReconstructed + C2);

    double ssim = numerator / denominator;

    return ssim;
}


// ============================================================
// FUNCTION: JPEG COMPRESS USING LIBJPEG-TURBO
//
// Input:
//      OpenCV BGR image
//
// Output:
//      Vector containing compressed JPEG bytes
// ============================================================

bool compressJPEG(
    const cv::Mat& image,
    vector<unsigned char>& compressedData,
    int quality)
{
    // Create TurboJPEG compressor
    tjhandle compressor = tjInitCompress();

    if (compressor == nullptr)
    {
        cerr << "ERROR: Could not initialize TurboJPEG compressor.\n";
        return false;
    }

    unsigned char* jpegBuffer = nullptr;
    unsigned long jpegSize = 0;

    // OpenCV stores color images as BGR.
    // TurboJPEG expects RGB when using TJPF_RGB.
    //
    // Therefore, convert BGR -> RGB.
    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

    int width = rgbImage.cols;
    int height = rgbImage.rows;

    int pixelFormat = TJPF_RGB;

    int pitch = width * 3;

    int result = tjCompress2(
        compressor,
        rgbImage.data,
        width,
        pitch,
        height,
        pixelFormat,
        &jpegBuffer,
        &jpegSize,
        TJSAMP_444,
        quality,
        TJFLAG_FASTDCT
    );

    if (result != 0)
    {
        cerr << "ERROR: JPEG compression failed: "
             << tjGetErrorStr()
             << "\n";

        tjDestroy(compressor);
        return false;
    }

    // Copy compressed data into our vector
    compressedData.assign(
        jpegBuffer,
        jpegBuffer + jpegSize
    );

    // Free memory allocated by TurboJPEG
    tjFree(jpegBuffer);

    // Destroy compressor
    tjDestroy(compressor);

    return true;
}


// ============================================================
// FUNCTION: JPEG DECOMPRESS USING LIBJPEG-TURBO
//
// Input:
//      JPEG compressed bytes
//
// Output:
//      Reconstructed OpenCV BGR image
// ============================================================

bool decompressJPEG(
    const vector<unsigned char>& compressedData,
    cv::Mat& reconstructed)
{
    // Create TurboJPEG decompressor
    tjhandle decompressor = tjInitDecompress();

    if (decompressor == nullptr)
    {
        cerr << "ERROR: Could not initialize TurboJPEG decompressor.\n";
        return false;
    }

    int width;
    int height;
    int subsampling;
    int colorspace;

    // Read JPEG header
    int result = tjDecompressHeader3(
        decompressor,
        compressedData.data(),
        compressedData.size(),
        &width,
        &height,
        &subsampling,
        &colorspace
    );

    if (result != 0)
    {
        cerr << "ERROR: Could not read JPEG header: "
             << tjGetErrorStr()
             << "\n";

        tjDestroy(decompressor);
        return false;
    }

    // Create RGB image
    cv::Mat rgbImage(
        height,
        width,
        CV_8UC3
    );

    // Decompress JPEG into RGB image
    result = tjDecompress2(
        decompressor,
        compressedData.data(),
        compressedData.size(),
        rgbImage.data,
        width,
        width * 3,
        height,
        TJPF_RGB,
        TJFLAG_FASTDCT
    );

    if (result != 0)
    {
        cerr << "ERROR: JPEG decompression failed: "
             << tjGetErrorStr()
             << "\n";

        tjDestroy(decompressor);
        return false;
    }

    // Convert RGB back to OpenCV BGR
    cv::cvtColor(
        rgbImage,
        reconstructed,
        cv::COLOR_RGB2BGR
    );

    // Destroy decompressor
    tjDestroy(decompressor);

    return true;
}


// ============================================================
// FUNCTION: Get all JPG/JPEG files from a folder
//
// Files are sorted alphabetically.
// ============================================================

vector<fs::path> getImageFiles(const fs::path& folder)
{
    vector<fs::path> files;

    if (!fs::exists(folder))
    {
        cerr << "ERROR: Folder does not exist: "
             << folder << "\n";

        return files;
    }

    // Read all files
    for (const auto& entry : fs::directory_iterator(folder))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        string extension =
            entry.path().extension().string();

        // Convert extension to lowercase
        transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            ::tolower
        );

        if (extension == ".jpg" ||
            extension == ".jpeg")
        {
            files.push_back(entry.path());
        }
    }

    // Sort alphabetically
    sort(
        files.begin(),
        files.end()
    );

    return files;
}


// ============================================================
// FUNCTION: Write CSV Header
// ============================================================

void writeCSVHeader(ofstream& csv)
{
    csv << "Class,"
        << "Image,"
        << "Original_Size_Bytes,"
        << "Compressed_Size_Bytes,"
        << "Compression_Ratio,"
        << "Space_Saving_Percent,"
        << "Compression_Time_ms,"
        << "Decompression_Time_ms,"
        << "Total_Time_ms,"
        << "MSE,"
        << "PSNR_dB,"
        << "SSIM\n";
}


// ============================================================
// FUNCTION: Write one image's results to CSV
// ============================================================

void writeImageResult(
    ofstream& csv,
    const ImageResult& result)
{
    csv << result.className << ","
        << result.imageName << ","
        << result.originalSize << ","
        << result.compressedSize << ","
        << result.compressionRatio << ","
        << result.spaceSaving << ","
        << result.compressionTime << ","
        << result.decompressionTime << ","
        << result.totalTime << ","
        << result.mse << ","
        << result.psnr << ","
        << result.ssim << "\n";
}


fs::path resolveProjectRoot()
{
    fs::path currentDir = fs::current_path();

    // First check the current working directory.
    if (fs::exists(currentDir / "natural_images"))
    {
        return currentDir;
    }

    // Then check the directory where the executable is located.
    fs::path executableDir =
        fs::absolute(fs::path(__FILE__)).parent_path();

    if (fs::exists(executableDir / "natural_images"))
    {
        return executableDir;
    }

    // Check the parent directory of src/
    fs::path projectRoot =
        executableDir.parent_path();

    if (fs::exists(projectRoot / "natural_images"))
    {
        return projectRoot;
    }

    // If not found, return current directory.
    return currentDir;
}


// ============================================================
// MAIN FUNCTION
// ============================================================

int main(int argc, char* argv[])
{
    // --------------------------------------------------------
    // DATASET PATH
    // --------------------------------------------------------
    //
    // Resolve the project root dynamically so paths work across
    // different working directories and machine locations.
    // --------------------------------------------------------

    fs::path projectRoot = resolveProjectRoot();
    fs::path datasetPath = projectRoot / "natural_images";


    // --------------------------------------------------------
    // OUTPUT DIRECTORIES
    // --------------------------------------------------------

    fs::path outputPath = projectRoot / "output";

    fs::path compressedPath =
        outputPath / "compressed";

    fs::path reconstructedPath =
        outputPath / "reconstructed";

    // CSV results are stored in the project-level results/ folder.
    fs::path resultsPath = projectRoot / "results";


    // Create output folders
    fs::create_directories(compressedPath);
    fs::create_directories(reconstructedPath);
    fs::create_directories(resultsPath);


    // --------------------------------------------------------
    // CHECK DATASET
    // --------------------------------------------------------

    if (!fs::exists(datasetPath))
    {
        cerr << "\nERROR: Dataset folder not found.\n";
        cerr << "Expected dataset path:\n";
        cerr << datasetPath << "\n";
        cerr << "\nPlease verify that natural_images exists in the project root.\n";

        return 1;
    }


    cout << "\n============================================\n";
    cout << " Sequential Image Compression Experiment\n";
    cout << "============================================\n";

    cout << "Dataset: "
         << datasetPath << "\n";

    cout << "JPEG Quality: "
         << JPEG_QUALITY << "\n";

    cout << "Classes: "
         << CLASS_NAMES.size() << "\n";


    // --------------------------------------------------------
    // SELECT EXPERIMENTS
    // --------------------------------------------------------
    //
    // No argument:
    //     Run 50, 100, 250 and 500 images per class.
    //
    // Argument 2:
    //     Run a small correctness test using exactly 2 images total.
    //
    // Argument 50 / 100 / 250 / 500:
    //     Run only that experiment (images per class).
    // --------------------------------------------------------

    bool testTwoTotal = false;
    vector<int> experiments;

    if (argc > 1)
    {
        try
        {
            int requestedCount = stoi(argv[1]);

            if (requestedCount == 2)
            {
                testTwoTotal = true;
                experiments.push_back(2);
            }
            else if (requestedCount == 50 ||
                     requestedCount == 100 ||
                     requestedCount == 250 ||
                     requestedCount == 500)
            {
                experiments.push_back(requestedCount);
            }
            else
            {
                cerr << "ERROR: Invalid image count.\n";
                cerr << "Use: 2, 50, 100, 250 or 500.\n";
                return 1;
            }
        }
        catch (...)
        {
            cerr << "ERROR: Invalid command-line argument.\n";
            cerr << "Use: 2, 50, 100, 250 or 500.\n";
            return 1;
        }
    }
    else
    {
        experiments = IMAGE_COUNTS;
    }

    for (int imagesPerClass : experiments)
    {
        cout << "\n\n--------------------------------------------\n";

        if (testTwoTotal)
        {
            cout << "Experiment: 2 images total (correctness test)\n";
            cout << "Total images: 2\n";
        }
        else
        {
            cout << "Experiment: "
                 << imagesPerClass
                 << " images/class\n";

            cout << "Total images: "
                 << imagesPerClass * CLASS_NAMES.size()
                 << "\n";
        }

        cout << "--------------------------------------------\n";


        // CSV file for this experiment
        string csvFileName =
            "sequential_" +
            to_string(imagesPerClass) +
            "_per_class.csv";

        fs::path csvPath =
            resultsPath / csvFileName;

        ofstream csv(csvPath);

        if (!csv.is_open())
        {
            cerr << "ERROR: Could not create CSV file.\n";
            continue;
        }

        // Write CSV header
        writeCSVHeader(csv);


        // ----------------------------------------------------
        // VARIABLES FOR OVERALL RESULTS
        // ----------------------------------------------------

        long long totalOriginalSize = 0;
        long long totalCompressedSize = 0;

        double totalCompressionTime = 0.0;
        double totalDecompressionTime = 0.0;
        double totalProcessingTime = 0.0;

        double totalMSE = 0.0;
        double totalPSNR = 0.0;
        double totalSSIM = 0.0;

        int successfulImages = 0;
        int requestedTotalImages = testTwoTotal ? 2 :
            imagesPerClass * static_cast<int>(CLASS_NAMES.size());


        // ----------------------------------------------------
        // PROCESS EACH CLASS
        // ----------------------------------------------------

        for (const string& className : CLASS_NAMES)
        {
            fs::path classPath =
                datasetPath / className;

            cout << "\nProcessing class: "
                 << className << "\n";


            // Get sorted image files
            vector<fs::path> imageFiles =
                getImageFiles(classPath);


            // Check if enough images exist
            if (imageFiles.size() < imagesPerClass)
            {
                cerr << "WARNING: "
                     << className
                     << " contains only "
                     << imageFiles.size()
                     << " images.\n";

                continue;
            }


            // ------------------------------------------------
            // SELECT IMAGES
            //
            // Normal experiments select the first N images from
            // every class. The 2-image correctness test selects
            // exactly 2 images total from the first class.
            //
            // Because files are alphabetically sorted, the first
            // images are airplane_0000.jpg, airplane_0001.jpg, etc.
            // ------------------------------------------------

            int imagesToProcess = testTwoTotal
                ? min(2 - successfulImages,
                      static_cast<int>(imageFiles.size()))
                : imagesPerClass;

            for (int i = 0;
                 i < imagesToProcess;
                 i++)
            {
                fs::path imagePath =
                    imageFiles[i];

                string imageName =
                    imagePath.filename().string();


                // ------------------------------------------------
                // READ ORIGINAL IMAGE
                // ------------------------------------------------

                cv::Mat original =
                    cv::imread(
                        imagePath.string(),
                        cv::IMREAD_COLOR
                    );

                if (original.empty())
                {
                    cerr << "Could not read: "
                         << imagePath << "\n";

                    continue;
                }


                // ------------------------------------------------
                // ORIGINAL FILE SIZE
                // ------------------------------------------------

                long long originalSize =
                    fs::file_size(imagePath);


                // ------------------------------------------------
                // CREATE EXPERIMENT-SPECIFIC OUTPUT DIRECTORIES
                // ------------------------------------------------

                fs::path compressedClassPath =
                    compressedPath /
                    to_string(imagesPerClass) /
                    className;

                fs::path reconstructedClassPath =
                    reconstructedPath /
                    to_string(imagesPerClass) /
                    className;

                fs::create_directories(
                    compressedClassPath
                );

                fs::create_directories(
                    reconstructedClassPath
                );


                // ------------------------------------------------
                // JPEG COMPRESSION
                // ------------------------------------------------

                vector<unsigned char> compressedData;


                auto compressionStart =
                    chrono::high_resolution_clock::now();


                bool compressionSuccess =
                    compressJPEG(
                        original,
                        compressedData,
                        JPEG_QUALITY
                    );


                auto compressionEnd =
                    chrono::high_resolution_clock::now();


                if (!compressionSuccess)
                {
                    cerr << "Compression failed for: "
                         << imageName << "\n";

                    continue;
                }


                double compressionTime =
                    chrono::duration<double, milli>(
                        compressionEnd -
                        compressionStart
                    ).count();


                // ------------------------------------------------
                // SAVE COMPRESSED JPEG
                // ------------------------------------------------

                fs::path compressedFilePath =
                    compressedClassPath /
                    imageName;

                ofstream compressedFile(
                    compressedFilePath,
                    ios::binary
                );

                compressedFile.write(
                    reinterpret_cast<const char*>(
                        compressedData.data()
                    ),
                    compressedData.size()
                );

                compressedFile.close();


                long long compressedSize =
                    compressedData.size();


                // ------------------------------------------------
                // JPEG DECOMPRESSION
                // ------------------------------------------------

                cv::Mat reconstructed;


                auto decompressionStart =
                    chrono::high_resolution_clock::now();


                bool decompressionSuccess =
                    decompressJPEG(
                        compressedData,
                        reconstructed
                    );


                auto decompressionEnd =
                    chrono::high_resolution_clock::now();


                if (!decompressionSuccess)
                {
                    cerr << "Decompression failed for: "
                         << imageName << "\n";

                    continue;
                }


                double decompressionTime =
                    chrono::duration<double, milli>(
                        decompressionEnd -
                        decompressionStart
                    ).count();


                double totalTime =
                    compressionTime +
                    decompressionTime;


                // ------------------------------------------------
                // SAVE RECONSTRUCTED IMAGE
                // ------------------------------------------------

                fs::path reconstructedFilePath =
                    reconstructedClassPath /
                    imageName;

                cv::imwrite(
                    reconstructedFilePath.string(),
                    reconstructed
                );


                // ------------------------------------------------
                // QUALITY METRICS
                // ------------------------------------------------

                double mse =
                    calculateMSE(
                        original,
                        reconstructed
                    );

                double psnr =
                    calculatePSNR(mse);

                double ssim =
                    calculateSSIM(
                        original,
                        reconstructed
                    );


                // ------------------------------------------------
                // COMPRESSION METRICS
                // ------------------------------------------------

                double compressionRatio =
                    static_cast<double>(
                        originalSize
                    )
                    /
                    static_cast<double>(
                        compressedSize
                    );

                double spaceSaving =
                    (
                        static_cast<double>(
                            originalSize -
                            compressedSize
                        )
                        /
                        originalSize
                    )
                    * 100.0;


                // ------------------------------------------------
                // STORE RESULT
                // ------------------------------------------------

                ImageResult result;

                result.className = className;
                result.imageName = imageName;

                result.originalSize =
                    originalSize;

                result.compressedSize =
                    compressedSize;

                result.compressionRatio =
                    compressionRatio;

                result.spaceSaving =
                    spaceSaving;

                result.compressionTime =
                    compressionTime;

                result.decompressionTime =
                    decompressionTime;

                result.totalTime =
                    totalTime;

                result.mse = mse;
                result.psnr = psnr;
                result.ssim = ssim;


                // Write per-image result
                csv << fixed << setprecision(6);

                writeImageResult(
                    csv,
                    result
                );


                // ------------------------------------------------
                // ADD TO OVERALL TOTALS
                // ------------------------------------------------

                totalOriginalSize +=
                    originalSize;

                totalCompressedSize +=
                    compressedSize;

                totalCompressionTime +=
                    compressionTime;

                totalDecompressionTime +=
                    decompressionTime;

                totalProcessingTime +=
                    totalTime;

                totalMSE += mse;

                totalPSNR += psnr;

                totalSSIM += ssim;

                successfulImages++;


                // Display progress
                cout << "\r"
                     << className
                     << ": "
                     << (i + 1)
                     << "/"
                     << imagesPerClass
                     << flush;
            }

            cout << "\n";

            if (testTwoTotal && successfulImages >= requestedTotalImages)
            {
                break;
            }
        }


        // --------------------------------------------------------
        // CALCULATE OVERALL METRICS
        // --------------------------------------------------------

        if (successfulImages > 0)
        {
            double overallCompressionRatio =
                static_cast<double>(
                    totalOriginalSize
                )
                /
                static_cast<double>(
                    totalCompressedSize
                );


            double overallSpaceSaving =
                (
                    static_cast<double>(
                        totalOriginalSize -
                        totalCompressedSize
                    )
                    /
                    totalOriginalSize
                )
                * 100.0;


            double averageMSE =
                totalMSE /
                successfulImages;


            double averagePSNR =
                totalPSNR /
                successfulImages;


            double averageSSIM =
                totalSSIM /
                successfulImages;


            // ----------------------------------------------------
            // WRITE OVERALL RESULT
            // ----------------------------------------------------

            csv << "\n";

            csv << "OVERALL,"
                << successfulImages << ","
                << totalOriginalSize << ","
                << totalCompressedSize << ","
                << overallCompressionRatio << ","
                << overallSpaceSaving << ","
                << totalCompressionTime << ","
                << totalDecompressionTime << ","
                << totalProcessingTime << ","
                << averageMSE << ","
                << averagePSNR << ","
                << averageSSIM
                << "\n";


            // ----------------------------------------------------
            // DISPLAY OVERALL RESULTS
            // ----------------------------------------------------

            cout << "\n============================================\n";
            cout << " Experiment Results\n";
            cout << "============================================\n";

            cout << "Images processed      : "
                 << successfulImages << "\n";

            cout << fixed << setprecision(4);

            cout << "Original size         : "
                 << totalOriginalSize / (1024.0 * 1024.0)
                 << " MB\n";

            cout << "Compressed size       : "
                 << totalCompressedSize / (1024.0 * 1024.0)
                 << " MB\n";

            cout << "Compression ratio     : "
                 << overallCompressionRatio
                 << "\n";

            cout << "Space saving          : "
                 << overallSpaceSaving
                 << " %\n";

            cout << "Compression time      : "
                 << totalCompressionTime
                 << " ms\n";

            cout << "Decompression time    : "
                 << totalDecompressionTime
                 << " ms\n";

            cout << "Total processing time : "
                 << totalProcessingTime
                 << " ms\n";

            cout << "Average MSE           : "
                 << averageMSE
                 << "\n";

            cout << "Average PSNR          : "
                 << averagePSNR
                 << " dB\n";

            cout << "Average SSIM          : "
                 << averageSSIM
                 << "\n";

            cout << "Results saved to      : "
                 << csvPath
                 << "\n";

            cout << "============================================\n";
        }

        csv.close();
    }


    // --------------------------------------------------------
    // PROGRAM COMPLETE
    // --------------------------------------------------------

    cout << "\n\nAll sequential experiments completed.\n";

    cout << "\nOutput structure:\n";

    cout << "Project folder/\n";
    cout << "├── compressed/\n";
    cout << "│   ├── 2/\n";
    cout << "│   ├── 50/\n";
    cout << "│   ├── 100/\n";
    cout << "│   ├── 250/\n";
    cout << "│   └── 500/\n";
    cout << "│\n";
    cout << "├── reconstructed/\n";
    cout << "│   ├── 2/\n";
    cout << "│   ├── 50/\n";
    cout << "│   ├── 100/\n";
    cout << "│   ├── 250/\n";
    cout << "│   └── 500/\n";
    cout << "│\n";
    cout << "results/\n";
    cout << "    ├── sequential_2_per_class.csv\n";
    cout << "    ├── sequential_50_per_class.csv\n";
    cout << "    ├── sequential_100_per_class.csv\n";
    cout << "    ├── sequential_250_per_class.csv\n";
    cout << "    └── sequential_500_per_class.csv\n";

    return 0;
}
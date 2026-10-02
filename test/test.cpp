#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <jpeglib.h>

#include <omp.h>

using namespace std;

int main()
{
    cout << "=====================================\n";
    cout << " Image Compression Library Test\n";
    cout << "=====================================\n\n";

    // --------------------------------------------------
    // TEST 1: OpenCV
    // --------------------------------------------------

    string imagePath = "test.jpg";

    cv::Mat image = cv::imread(imagePath);

    if (image.empty())
    {
        cout << "[OpenCV] FAILED\n";
        cout << "Could not read: " << imagePath << "\n";
        cout << "Place a JPG image named test.jpg in this folder.\n";
        return 1;
    }

    cout << "[OpenCV] PASSED\n";
    cout << "Image size: "
         << image.cols << " x " << image.rows << "\n";
    cout << "Channels: " << image.channels() << "\n\n";


    // --------------------------------------------------
    // TEST 2: OpenMP
    // --------------------------------------------------

    int maxThreads = omp_get_max_threads();

    cout << "[OpenMP] Maximum available threads: "
         << maxThreads << "\n";

    #pragma omp parallel
    {
        #pragma omp single
        {
            cout << "[OpenMP] Threads currently running: "
                 << omp_get_num_threads() << "\n";
        }
    }

    cout << "[OpenMP] PASSED\n\n";


    // --------------------------------------------------
    // TEST 3: libjpeg
    // --------------------------------------------------

    vector<uchar> jpegBuffer;

    vector<int> parameters;
    parameters.push_back(cv::IMWRITE_JPEG_QUALITY);
    parameters.push_back(75);

    bool encoded = cv::imencode(
        ".jpg",
        image,
        jpegBuffer,
        parameters
    );

    if (!encoded)
    {
        cout << "[JPEG] FAILED\n";
        return 1;
    }

    cout << "[JPEG] Compression PASSED\n";
    cout << "Original image memory: "
         << image.total() * image.elemSize()
         << " bytes\n";

    cout << "Compressed JPEG memory: "
         << jpegBuffer.size()
         << " bytes\n";


    // --------------------------------------------------
    // JPEG DECOMPRESSION
    // --------------------------------------------------

    cv::Mat reconstructed =
        cv::imdecode(jpegBuffer, cv::IMREAD_COLOR);

    if (reconstructed.empty())
    {
        cout << "[JPEG] Decompression FAILED\n";
        return 1;
    }

    cout << "[JPEG] Decompression PASSED\n";


    // --------------------------------------------------
    // Calculate MSE
    // --------------------------------------------------

    cv::Mat originalFloat;
    cv::Mat reconstructedFloat;

    image.convertTo(originalFloat, CV_32F);
    reconstructed.convertTo(reconstructedFloat, CV_32F);

    cv::Mat difference =
        originalFloat - reconstructedFloat;

    cv::Mat squaredDifference;

    cv::multiply(
        difference,
        difference,
        squaredDifference
    );

    double mse =
        cv::sum(squaredDifference)[0] /
        (image.total() * image.channels());


    // --------------------------------------------------
    // Calculate PSNR
    // --------------------------------------------------

    double psnr;

    if (mse == 0)
    {
        psnr = INFINITY;
    }
    else
    {
        psnr =
            10.0 *
            log10((255.0 * 255.0) / mse);
    }

    cout << "\nMSE  : " << mse << "\n";
    cout << "PSNR : " << psnr << " dB\n";


    // --------------------------------------------------
    // Save reconstructed image
    // --------------------------------------------------

    bool saved = cv::imwrite(
        "reconstructed.jpg",
        reconstructed,
        parameters
    );

    if (saved)
    {
        cout << "\nReconstructed image saved as:\n";
        cout << "reconstructed.jpg\n";
    }


    // --------------------------------------------------
    // Final result
    // --------------------------------------------------

    cout << "\n=====================================\n";
    cout << " All tests completed successfully!\n";
    cout << "=====================================\n";

    return 0;
}
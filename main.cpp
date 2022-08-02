#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;

struct ThresholdData {
    int type = 3;
    int value = 0;
};

int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_binary_value = 255;

Mat src_gray, dst;
const char* window_name = "Threshold Demo";
const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

void Thresholding_Operations(int slider, void* values) {
    threshold(src_gray, dst, threshold_value, max_binary_value, threshold_type);
    imshow(window_name, dst);
}

// 0.2 pi

int main() {
    std::string image_path = samples::findFile("playing-cards.png");
    src_gray = imread(image_path, IMREAD_GRAYSCALE);

    if(src_gray.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    namedWindow(window_name, WINDOW_AUTOSIZE);
    // imshow("Display window", img);
    createTrackbar( trackbar_type,
                    window_name, &threshold_type,
                    max_type, Thresholding_Operations ); // Create a Trackbar to choose type of Threshold
    createTrackbar( trackbar_value,
                    window_name, &threshold_value,
                    max_value, Thresholding_Operations ); // Create a Trackbar to choose Threshold value
    Thresholding_Operations( 0, 0 ); // Call the function to initialize

    waitKey();
    return 0;
}


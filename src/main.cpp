#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include "EnhancedWindow.h"

#define WINDOW_NAME    "CVUI Enhanced Window Component"


int main() 
{   
	// "Frame buffer"
	cv::Mat frame = cv::Mat(720, 1280, CV_8UC3);
	cv::Mat cards_original = cv::imread("playing-cards.png");
	cv::Mat cards = cards_original.clone();

	int low_threshold = 50; 
	int high_threshold = 150;
	bool use_canny = false;


	// Create a settings window using the EnhancedWindow class.
	// int x, int y, int width, int height, const cv::String& title, bool minimizable = true, double theFontScale = cvui::DEFAULT_FONT_SCALE
	EnhancedWindow settings(10, 50, 270, 180, "Settings");
	EnhancedWindow image(0, 0, cards.cols + 20, cards.rows + 40, "Source Image");
	int image_width = cards.cols;
	int image_height = cards.rows;
	double scale = 1.0;
	// Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
	cvui::init(WINDOW_NAME);

	while (true) 
	{
		// Clear color
		frame = cv::Scalar(49, 52, 49);
		
		// Resize windows
		int newHeight = std::min(frame.rows, image_height + 100);
		int newWidth = std::min(frame.cols, image_width + 20);

		image.setHeight(newHeight);
		image.setWidth(newWidth);

		cv::resize(cards_original, cards, { newWidth - 20, newHeight - 100});

		// Render the settings window and its content, if it is not minimized.
		settings.begin(frame);
		if (!settings.isMinimized()) {
			cvui::checkbox("Use Canny Edge", &use_canny);
			cvui::trackbar(settings.width() - 20, &low_threshold, 5, 150);
			cvui::trackbar(settings.width() - 20, &high_threshold, 80, 300);
			cvui::space(20); // add 20px of empty space
			cvui::text("Drag and minimize this settings window", 0.4, 0xff0000);
		}
		settings.end();
		
		image.begin(frame);
		if (!image.isMinimized()){
			cvui::beginRow(-1, -1, 20);
				cvui::trackbar(100, &image_width, 220, frame.cols);
				cvui::trackbar(100, &image_height, 100, frame.rows);
			cvui::endRow();
			cvui::space(5);
			cvui::image(cards);
		}
		image.end();

		// Update all cvui internal stuff, e.g. handle mouse clicks, and show
		// everything on the screen.
		cvui::imshow(WINDOW_NAME, frame);

		// Check if ESC was pressed
		if (cv::waitKey(30) == 27) {
			break;
		}
	}

	return 0;
}


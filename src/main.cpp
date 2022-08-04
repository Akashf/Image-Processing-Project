#include <iostream>
#include <unordered_map>
#include <array>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"

#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include "EnhancedWindow.h"

#define WINDOW_NAME    "CVUI Enhanced Window Component"


struct CannyParameters
{
	int low_threshold = 50;
	int high_threshold = 150;
};

struct GaussianParameters
{
	int kernel_size;
	int sigma;
};

struct ThresholdParameters
{
	int threshold = 100;
	int max = 255;
	int type = cv::THRESH_BINARY;
};


int main() 
{   
	// OpenCV config
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

	// "Frame buffer"
	int window_height = 720;
	int window_width = 1280;
	cv::Size rank_size(30, 45);

	cv::Mat frame = cv::Mat(window_height, window_width, CV_8UC3);

	// Source image 
	cv::Mat source = cv::imread("cards-scuffed.jpg", cv::IMREAD_GRAYSCALE);

	// Load rank and suit templates
	std::vector<std::pair<std::string, cv::Mat>> rank_images = {};
	std::vector<std::string> rank_names = {
		"Ace", "Two", "Three", "Four", "Five", "Six",
		"Seven", "Eight", "Nine", "Ten", "Jack", "Queen",
		"King"
	};
	for (const auto& rank: rank_names)
	{
		cv::Mat img = cv::imread("images/" + rank + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat resized;
		cv::resize(img, resized, rank_size);
		rank_images.push_back({ rank, resized});
	}

	std::vector<std::pair<std::string, cv::Mat>> suit_images = {};
	std::vector<std::string> suit_names = {
		"Hearts", "Clubs", "Spades", "Diamonds"
	};

	for (const auto& suit: suit_names)
	{
		suit_images.push_back({suit, cv::imread("images/" + suit + ".png", cv::IMREAD_GRAYSCALE) });
	}

	// Image to display 
	cv::Mat cards = source.clone();

	// Gaussian Parameters 
	GaussianParameters gauss_params; 
	gauss_params.kernel_size = 5;
	gauss_params.sigma = 3;

	// Canny parameters 
	CannyParameters canny_params; 
	canny_params.low_threshold = 50;
	canny_params.high_threshold = 150;

	// Threshold
	ThresholdParameters threshold_params;

	int active_image_index = 0;
	std::string active_stage = "Blurred";
	
	// Create a settings window using the EnhancedWindow class.
	// int x, int y, int width, int height, const cv::String& title, bool minimizable = true, double theFontScale = cvui::DEFAULT_FONT_SCALE
	EnhancedWindow settings(0, 0, 320, window_height, "Settings");
	EnhancedWindow image(settings.width(), 0, cards.cols + 20, cards.rows + 40, "Active Image");
	int image_width = cards.cols;
	int image_height = cards.rows;
	double scale = 1.0;

	// Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
	cvui::init(WINDOW_NAME);
	cv::Mat new_cards;
	auto time = std::chrono::high_resolution_clock::now();

	std::vector<std::string> stage_titles = {
		"Source",
		"Blurred",
		"Equalized",
		"Edges",
		"Contours",
	};

	std::unordered_map<std::string, cv::Mat> pipe_out = {
		{"Source", cv::Mat()},
		{"Blurred", cv::Mat()},
		{"Equalized", cv::Mat()},
		{"Binarized", cv::Mat()},
		{"Edges", cv::Mat()},
		{"Contours", cv::Mat()},
	};

	cv::Mat display_image; 
	bool save_image = false;
	while (true) 
	{
		// FPS Tracking 
		auto now = std::chrono::high_resolution_clock::now();
		auto time_delta = now - time;
		time = now;
		auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(time_delta).count() / 1000.0;
		std::cout << "Frame time (ms): " << frame_time << "\n";
	
		// Clear background color
		frame = cv::Scalar(53, 101, 77);

		if (save_image)
		{
			cv::imwrite("image_out.png", pipe_out[active_stage]);
			save_image = false;
		}

		// Resize image window
		int newHeight = std::min(frame.rows, image_height + 100);
		int newWidth = std::min(frame.cols, image_width + 20);
		image.setHeight(newHeight);
		image.setWidth(newWidth);
		cv::resize(source, cards, { newWidth - 20, newHeight - 100 });

		// Instantiate and execute pipeline 
		cv::GMat g_in;
		cv::GMat g_blurred = cv::gapi::gaussianBlur(g_in, { gauss_params.kernel_size, gauss_params.kernel_size }, gauss_params.sigma);
		cv::GMat g_equalized = cv::gapi::equalizeHist(g_blurred);
		cv::GMat g_edges = cv::gapi::Canny(g_equalized, canny_params.low_threshold, canny_params.high_threshold);
		cv::GArray<cv::GArray<cv::Point>> g_contours = cv::gapi::findContours(g_edges, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		cv::GComputation pipeline(cv::GIn(g_in), cv::GOut(g_blurred, g_equalized, g_edges, g_contours));
	
		// Execute pipeline
		pipe_out["Source"] = cards;
		std::vector<std::vector<cv::Point>> contours;
		pipeline.apply(
			cv::gin(cards), 
			cv::gout
			(
				pipe_out["Blurred"],
				pipe_out["Equalized"],
				pipe_out["Edges"],
				contours
			) 
		);

		// Generate contour overlay
		cv::Mat contour_base = cv::Mat::zeros(cards.size(), CV_8UC3);
		for (size_t i = 0; i < contours.size(); i++)
		{
			cv::drawContours(contour_base, contours, i, cv::Scalar(255, 0, 0), 1);
		}
		pipe_out["Contours"] = contour_base.clone();

		
		std::vector<cv::Rect> boundRect(contours.size());
		std::vector<cv::Mat> card_images = {};
		std::vector<cv::Point2f> target_pts = {{0, 0}, {0, 349}, {249, 349}, {249, 0}};

		size_t i = 0;
		for (auto& c: contours)
		{
			std::vector<cv::Point2f> output;
			float e = 0.01 * cv::arcLength(c, true);
			cv::approxPolyDP(c, output, e, true);

			if (output.size() != 4) {
				continue; 
			}

			float x_sum = 0;
			float y_sum = 0;
			for (auto p: output)
			{
				x_sum += p.x;
				y_sum += p.y;
			}
			cv::Point2f mid(x_sum / 4, y_sum / 4);

			// Determine semantic location of this point in image
			std::vector<cv::Point2f> src(4);
			for (auto p: output)
			{
				cv::Point2f delta = mid - p;

				if (delta.x > 0 && delta.y > 0) 
				{
					// Top left
					src[0] = p;
				}
				else if (delta.x < 0 && delta.y > 0) 
				{
					// Top right
					src[3] = p;
				}
				else if (delta.x > 0 && delta.y < 0)
				{
					// Bottom left
					src[1] = p;
				}
				else
				{
					// Bottom right
					src[2] = p;
				}
			}

			cv::Mat p = cv::getPerspectiveTransform(src, target_pts);
			cv::Mat img;

			cv::warpPerspective(cards, img, p, cv::Size(250, 350));
			card_images.push_back(img); 

			std::stringstream ss; 
			ss << "Card: " << i++;

			cv::imshow(ss.str(), img);
		}

		// For each image
		// Extract + identify rank

		i = 0;
		for (auto& img: card_images)
		{
			// (45,60) 2
			// (40, 60) Q
			// (40, 60) 10
			// (35, 55) K 
			// (40, 55) A
			// (45, 55)
			cv::Mat sub_image = img(cv::Range(10, 55), cv::Range(10, 40));
	
			cv::Mat blurred;
			cv::GaussianBlur(sub_image, blurred, cv::Size(5,5), 0);

			/*cv::Mat thresholded;
			cv::adaptiveThreshold(sub_image, thresholded, 255, cv::THRESH_BINARY_INV, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 11, 10);*/

			cv::Mat thresholded;
			cv::threshold(blurred, thresholded, 200, 255, cv::THRESH_BINARY_INV);

			cv::Mat eroded;
			auto element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5,5));
			cv::erode(thresholded, eroded, element);

			std::vector<std::vector<cv::Point>> contours; 
			cv::findContours(eroded, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			//// Bound based on centroid of the image 
			//std::vector<cv::Moments> moments; 
			//for (auto& c: contours)
			//{
			//	moments.push_back(cv::moments(c, true));
			//}

			//// Calculate centroid
			//std::vector<cv::Point2f> centroids(contours.size());
			//for (int i = 0; i < contours.size(); i++)
			//{
			//	centroids[i] = cv::Point2f(moments[i].m10 / moments[i].m00, moments[i].m01 / moments[i].m00);
			//}

			// Select largest contour
			std::vector<cv::Point> largest_c;

			float max_area = 0;
			for (auto& c: contours)
			{
				float area = cv::contourArea(c);
				if (area > max_area)
				{
					max_area = area;
					largest_c = c;
				}
			}

			// Draw bounding box of largest contour
			cv::Rect bb = cv::boundingRect(largest_c);
			cv::Mat base = cv::Mat::zeros(sub_image.size(), CV_8UC3);
			for (int i = 0; i < contours.size(); i++)
			{
				cv::drawContours(base, contours, i, { 255, 0, 0 }, 1);
			}

			cv::Mat boundedImage(eroded, bb);

			boundedImage = ~boundedImage;

			std::stringstream ss;
			ss << "Contours: " << i++;
			// cv::imshow(ss.str(), base);

			int min_diff = std::numeric_limits<int>().max();
			float max_conf = 0;
			std::string best_match = "";

			for (const auto& img: rank_images)
			{
				cv::Mat diff_image; 
				cv::Mat tem;
				cv::resize(img.second, tem, boundedImage.size());
				cv::absdiff(boundedImage, tem, diff_image);
				int avg_diff = cv::sum(diff_image)[0] / 255;
				if (avg_diff < min_diff)
				{
					min_diff = avg_diff;
					best_match = img.first;
				}	
			}

			ss.clear();
			ss << "Rank: " << best_match << " " << i++;

			// cv::imshow(ss.str(), boundedImage);
		}

		// Select active stage 
		active_stage = stage_titles[active_image_index];
		display_image = pipe_out[active_stage];

		// Render the settings window and its content, if it is not minimized.
		settings.begin(frame);
		if (!settings.isMinimized()) 
		{
			int width = 300;
			cvui::text("Pipeline Stage - " + active_stage);
			cvui::trackbar(width, &active_image_index, 0, (int)stage_titles.size() - 1, 1, "%0.1d", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);

			cvui::text("Save Current Image");
			cvui::space(8);
			save_image = cvui::button("Save");
			cvui::space(10);

			cvui::text("Gaussian Configuration");
			cvui::space(10);
			
			cvui::text("Kernel Size");
			cvui::space(10);
			cvui::trackbar(width, &gauss_params.kernel_size, 1, 9, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 2);

			cvui::text("Sigma");
			cvui::space(10);
			cvui::trackbar(width, &gauss_params.sigma, 0, 10, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10); // add 20px of empty space

			cvui::text("Threshold Configuration");
			cvui::space(10);
			cvui::text("Threshold");
			cvui::space(4);
			cvui::trackbar(width, &threshold_params.threshold, 0, 255, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);

			cvui::text("Canny Configuration");
			cvui::space(10);
			cvui::text("Low Threshold");
			cvui::space(4);
			cvui::trackbar(width, &canny_params.low_threshold, 0, 255, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(4);
			cvui::text("High Threshold");
			cvui::space(4);
			cvui::trackbar(width, &canny_params.high_threshold, 0, 255, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);
		}
		settings.end();
		
		image.begin(frame);
		if (!image.isMinimized())
		{
			cv::Mat disp; 
			if (active_stage == "Contours")
			{
				disp = display_image;
			}
			else
			{
				cv::cvtColor(display_image, disp, cv::COLOR_GRAY2BGR);
			}
			
			cvui::beginRow(-1, -1, 20);
				cvui::trackbar(100, &image_width, 220, frame.cols);
				cvui::trackbar(100, &image_height, 100, frame.rows);
			cvui::endRow();
			cvui::space(5);
			cvui::image(disp);
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


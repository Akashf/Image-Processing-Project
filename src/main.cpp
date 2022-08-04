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
	int low_threshold = 0;
	int high_threshold = 255;
};

struct GaussianParameters
{
	int kernel_size = 3;
	int sigma = 0;
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
	int window_height = 1080;
	int window_width = 1920;
	cv::Size rank_size(30, 45);

	cv::Mat frame = cv::Mat(window_height, window_width, CV_8UC3);

	// Source image 
	cv::Mat source = cv::imread("cards.jpg");

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
	cv::Mat cards_color = source;

	// Gaussian Parameters 
	GaussianParameters gauss_params; 
	gauss_params.kernel_size = 3;
	gauss_params.sigma = 0;

	// Canny parameters 
	CannyParameters canny_params; 
	canny_params.low_threshold = 0;
	canny_params.high_threshold = 255;

	// Threshold
	ThresholdParameters threshold_params;

	int active_image_index = 0;
	std::string active_stage = "Blurred";
	
	// Create a settings window using the EnhancedWindow class.
	// int x, int y, int width, int height, const cv::String& title, bool minimizable = true, double theFontScale = cvui::DEFAULT_FONT_SCALE
	EnhancedWindow settings(0, 0, 320, window_height, "Settings");
	EnhancedWindow image(settings.width(), 0, cards_color.cols + 20, cards_color.rows + 40, "Active Image");
	int image_width = cards_color.cols;
	int image_height = cards_color.rows;
	double scale = 1.0;

	// Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
	cvui::init(WINDOW_NAME);
	cv::Mat new_cards;
	auto time = std::chrono::high_resolution_clock::now();

	std::vector<std::string> stage_titles = {
		"Source",
		"Blurred",
		// "Equalized",
		"Edges",
		"Contours",
		"Rectangle Contours",
		"Output"
	};

	std::unordered_map<std::string, cv::Mat> pipe_out = {
		{"Source", cv::Mat()},
		{"Blurred", cv::Mat()},
		{"Equalized", cv::Mat()},
		{"Binarized", cv::Mat()},
		{"Edges", cv::Mat()},
		{"Contours", cv::Mat()},
		{"Rectangle Contours", cv::Mat()},
		{"Output", cv::Mat()},
	};

	std::string best_match = "";

	cv::Mat display_image; 
	bool save_image = false;
	cv::Mat cam_frame; 
	cv::VideoCapture camera(0);

	bool camera_available = true;
	if (!camera.isOpened())
	{
		std::cout << "Cannot connect to camera";
		camera_available = false;
	}

	bool use_camera = false;
	cv::Mat cards;
	while (true) 
	{
		// FPS Tracking 
		auto now = std::chrono::high_resolution_clock::now();
		auto time_delta = now - time;
		time = now;
		auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(time_delta).count() / 1000.0;
		std::cout << "Frame time (ms): " << frame_time << "\n";
	
		// Read image from camera
		if (use_camera)
		{
			camera >> cam_frame;
			cards_color = cam_frame;
		}
		else
		{
			cards_color = source.clone();
		}

		cv::cvtColor(cards_color, cards, cv::COLOR_BGR2GRAY);

		// Clear background color
		frame = cv::Scalar(53, 101, 77);
		if (save_image)
		{
			cv::imwrite("image_out.png", pipe_out[active_stage]);
			save_image = false;
		}

		// Resize image window to fit camera frame 
		int newHeight = cards.rows + 40;
		int newWidth = cards.cols + 20;

		image.setHeight(newHeight);
		image.setWidth(newWidth);

		// Instantiate and execute pipeline 
		cv::GMat g_in;
		cv::GMat g_blurred = cv::gapi::gaussianBlur(g_in, { gauss_params.kernel_size, gauss_params.kernel_size }, gauss_params.sigma);
		cv::GMat g_equalized = cv::gapi::equalizeHist(g_blurred);
		cv::GMat g_edges = cv::gapi::Canny(g_blurred, canny_params.low_threshold, canny_params.high_threshold);
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

		std::vector<cv::Rect> boundRect(contours.size());
		std::vector<cv::Mat> card_images = {};
		std::vector<cv::Point> card_midpoints = {};
		std::vector<std::string> card_best_guesses = {};
		std::vector<std::string> suit_best_guesses = {};
		std::vector<cv::Point2f> target_pts = {{0, 0}, {0, 349}, {249, 349}, {249, 0}};
		std::vector<std::vector<cv::Point>> rect_contours = {};

		size_t i = 0;
		for (auto& c: contours)
		{
			std::vector<cv::Point2f> output;
			std::vector<cv::Point> output_i;

			float e = 0.01 * cv::arcLength(c, true);
			cv::approxPolyDP(c, output, e, true);

			if (output.size() != 4 || cv::contourArea(output) < 5000) {
				continue; 
			}

			float x_sum = 0;
			float y_sum = 0;
			for (auto p: output)
			{
				output_i.push_back(cv::Point((int)p.x, (int)p.y));

				x_sum += p.x;
				y_sum += p.y;
			}
			cv::Point2f mid(x_sum / 4, y_sum / 4);
			card_midpoints.push_back(mid);
			rect_contours.push_back(output_i);

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
		}

		// Extract + identify rank
		i = 0;
		for (auto& img: card_images)
		{
			// Extract + Identify Rank
			cv::Mat rank_image = img(cv::Range(0, 55), cv::Range(0, 40));
	
			cv::Mat rank_thresholded;
			cv::threshold(rank_image, rank_thresholded, 150, 255, cv::THRESH_OTSU);
			rank_thresholded = ~rank_thresholded;

			cv::Mat rank_dilated;
			auto element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(4,4));
			cv::dilate(rank_thresholded, rank_dilated, element);

			std::vector<std::vector<cv::Point>> contours; 
			cv::findContours(rank_dilated, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

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
			cv::Mat base = cv::Mat::zeros(rank_image.size(), CV_8UC3);
			for (int i = 0; i < contours.size(); i++)
			{
				cv::drawContours(base, contours, i, { 255, 0, 0 }, 1);
			}

			cv::Mat bounded_rank = cv::Mat::zeros(rank_image.size(), CV_8UC1);
			if (!largest_c.empty())
			{
				bounded_rank = rank_dilated(bb);
			}

			cv::Mat bounded_eroded = cv::Mat::zeros(bounded_rank.size(), CV_8UC1);
			auto element1 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(8, 8));
			cv::erode(bounded_rank, bounded_eroded, element1);
			
			cv::Mat bounded_dilated;
			element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(8, 8));
			cv::dilate(bounded_eroded, bounded_dilated, element1);

			bounded_dilated = ~bounded_dilated;
			bounded_rank = ~bounded_rank;
			bounded_eroded = ~bounded_eroded;

			cv::Mat rank_identity = bounded_dilated;

			int min_diff = std::numeric_limits<int>().max();
			float max_conf = 0;
	
			for (const auto& img: rank_images)
			{
				cv::Mat diff_image; 
				cv::Mat tem;
				cv::resize(img.second, tem, rank_identity.size());
				cv::absdiff(rank_identity, tem, diff_image);
				int avg_diff = cv::sum(diff_image)[0] / 255;
				if (avg_diff < min_diff)
				{
					min_diff = avg_diff;
					best_match = img.first;
				}	
			}

			std::stringstream ss;
			ss << "Rank Best Guess: " << best_match;
			card_best_guesses.push_back(best_match);
		
			// Extract + Identify Suit 
			cv::Mat suit_image = img(cv::Range(55, 100), cv::Range(0, 40));
		
			cv::Mat suit_thresholded; 
			cv::threshold(suit_image, suit_thresholded, 120, 255, cv::THRESH_OTSU);
			suit_thresholded = ~suit_thresholded;

			cv::Mat suit_eroded;
			element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
			cv::erode(suit_thresholded, suit_eroded, element);
			
			cv::Mat suit_dilated;
			element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
			cv::dilate(suit_eroded, suit_dilated, element);

			std::vector<std::vector<cv::Point>> suit_contours;
			cv::findContours(suit_dilated, suit_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			// Calculate bb of largest area contour
			cv::Rect suit_bb; 
			{
				std::vector<cv::Point> largest_contour;

				float max_area = 0;
				for (auto& c : suit_contours)
				{
					float area = cv::contourArea(c);
					if (area > max_area)
					{
						max_area = area;
						largest_contour = c;
					}
				}
				suit_bb = cv::boundingRect(largest_contour);
			}

			cv::Mat bounded_suit = suit_dilated.clone();
			if (!suit_bb.empty())
			{
				bounded_suit = suit_dilated(suit_bb);
			}

			// Final suit 
			bounded_suit = ~bounded_suit;

			// Identify rank 
			min_diff = std::numeric_limits<int>().max();
			std::string suit_match = "";
			for (const auto& img : suit_images)
			{
				cv::Mat diff_image;
				cv::Mat tem;
				cv::resize(img.second, tem, bounded_suit.size());
				cv::absdiff(bounded_suit, tem, diff_image);
				int avg_diff = cv::sum(diff_image)[0] / 255;
				if (avg_diff < min_diff)
				{
					min_diff = avg_diff;
					suit_match = img.first;
				}
			}

			suit_best_guesses.push_back(suit_match);
		}

		// Generate original contour overlay
		cv::Mat contour_base;
		cv::cvtColor(cards, contour_base, cv::COLOR_GRAY2BGR);

		for (size_t i = 0; i < contours.size(); i++)
		{
			cv::drawContours(contour_base, contours, i, cv::Scalar(0, 0, 255), 4);
		}
		pipe_out["Contours"] = contour_base.clone();

		// Generate rectangle contour 
		cv::Mat rect_contour_base;
		cv::cvtColor(cards, rect_contour_base, cv::COLOR_GRAY2BGR);

		for (size_t i = 0; i < rect_contours.size(); i++)
		{
			cv::drawContours(rect_contour_base, rect_contours, i, cv::Scalar(0, 0, 255), 2);

		}
		pipe_out["Rectangle Contours"] = rect_contour_base.clone();

		for (size_t i = 0; i < rect_contours.size(); i++)
		{
			cv::drawContours(cards_color, rect_contours, i, cv::Scalar(0, 0, 255), 2);

		}

		pipe_out["Output"] = cards_color.clone();
		// Draw best match rank and suit at center of image 
		for (size_t i = 0; i < card_images.size(); i++)
		{
			auto mid = card_midpoints[i];
			std::string rank_best_guess = card_best_guesses[i];
			std::string suit_best_guess = suit_best_guesses[i];
			
			cv::Size rank_size = cv::getTextSize(rank_best_guess, cv::FONT_HERSHEY_COMPLEX, 1, 2, nullptr);
			cv::Point rank_origin = cv::Point(mid.x - rank_size.width / 2, mid.y + rank_size.height / 2);

			cv::Size suit_size = cv::getTextSize(suit_best_guess, cv::FONT_HERSHEY_COMPLEX, 0.75, 2, nullptr);
			cv::Point suit_origin = cv::Point(mid.x - suit_size.width / 2, mid.y + suit_size.height / 2);

			cv::putText(pipe_out["Output"], rank_best_guess, rank_origin, cv::FONT_HERSHEY_COMPLEX, 1.0, CV_RGB(0, 0, 255), 2);
			cv::putText(pipe_out["Output"], suit_best_guess, suit_origin + cv::Point(0, 24), cv::FONT_HERSHEY_COMPLEX, 0.75, CV_RGB(0, 0, 255), 2);
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

			if (camera_available)
			{
				cvui::text("Save Current Image");
				cvui::space(8);
				cvui::checkbox("Live", &use_camera);
				cvui::space(10);
			}
			
			cvui::text("Gaussian Configuration");
			cvui::space(10);
			
			cvui::text("Kernel Size");
			cvui::space(10);
			cvui::trackbar(width, &gauss_params.kernel_size, 1, 9, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 2);

			cvui::text("Sigma");
			cvui::space(10);
			cvui::trackbar(width, &gauss_params.sigma, 0, 10, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10); // add 20px of empty space

			/*cvui::text("Threshold Configuration");
			cvui::space(10);
			cvui::text("Threshold");
			cvui::space(4);
			cvui::trackbar(width, &threshold_params.threshold, 0, 255, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);*/

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
			if (active_stage == "Contours" || active_stage == "Output" || active_stage == "Rectangle Contours")
			{
				disp = display_image;
			}
			else
			{
				cv::cvtColor(display_image, disp, cv::COLOR_GRAY2BGR);
			}
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


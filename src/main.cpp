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

#define WINDOW_NAME    "Most Constrained Card Detector"


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
	const std::string test_image_root = "assets/test_images/";
	cv::Mat source = cv::imread(test_image_root + "cards-numerous.jpg");

	// Load rank and suit templates
	std::vector<std::pair<std::string, cv::Mat>> rank_images = {};
	std::vector<std::string> rank_names = {
		"Ace", "Two", "Three", "Four", "Five", "Six",
		"Seven", "Eight", "Nine", "Ten", "Jack", "Queen",
		"King"
	};

	const std::string template_root = "assets/template_images/";
	for (const auto& rank: rank_names)
	{
		cv::Mat img = cv::imread(template_root + rank + ".png", cv::IMREAD_GRAYSCALE);
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
		suit_images.push_back({suit, cv::imread(template_root + suit + ".png", cv::IMREAD_GRAYSCALE) });
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
	
	// Create windows
	EnhancedWindow settings(0, 0, 320, window_height, "Settings");
	EnhancedWindow image(settings.width(), 0, cards_color.cols + 20, cards_color.rows + 40, "Active Image");
	EnhancedWindow sub_image(settings.width(), window_height - 500, 350, 500, "Individual Card View");

	// Create image stores
	std::vector<std::string> sub_stage_titles = {
		"Warped",
		"Rank",
		"Rank Threshold",
		"Rank Dilated",
		"Rank Contours",
		"Rank Bounded",
		"Suit",
		"Suit Threshold",
		"Suit Dilated",
		"Suit Eroded",
		"Suit Contours",
		"Suit Bounded",
	};

	std::unordered_map<std::string, cv::Mat> card_img_data = {
		{ "Warped", cv::Mat::zeros(10, 10, CV_8UC3) },
		{ "Rank", cv::Mat::zeros(10, 10, CV_8UC1)},
		{ "Rank Thresholded", cv::Mat::zeros(10, 10, CV_8UC1)},
		{ "Rank Dilated", cv::Mat::zeros(10, 10, CV_8UC1)},
		{ "Rank Contours", cv::Mat::zeros(10, 10, CV_8UC3)},
		{ "Rank Bounded", cv::Mat::zeros(10, 10, CV_8UC1)},
		{ "Rank Final", cv::Mat::zeros(10, 10, CV_8UC1)},
		{ "Suit", cv::Mat::zeros(10, 10, CV_8UC1)},
		{ "Suit Thresholded", cv::Mat::zeros(10, 10, CV_8UC1)},
		{ "Suit Eroded", cv::Mat::zeros(10, 10, CV_8UC1)},
		{ "Suit Dilated", cv::Mat::zeros(10, 10, CV_8UC1)},
		{ "Suit Countours", cv::Mat::zeros(10, 10, CV_8UC3)},
		{ "Suit Bounded", cv::Mat::zeros(10, 10, CV_8UC1)}
	};

	std::vector<std::string> stage_titles = {
		"Source",
		"Blurred",
		"Equalized",
		"Edges",
		"Contours",
		"Rectangle Contours",
		"Output"
	};

	std::unordered_map<std::string, cv::Mat> pipe_out = {
		{"Source", cv::Mat::zeros(10, 10, CV_8UC1)},
		{"Blurred", cv::Mat()},
		{"Equalized", cv::Mat()},
		{"Binarized", cv::Mat()},
		{"Edges", cv::Mat()},
		{"Contours", cv::Mat()},
		{"Rectangle Contours", cv::Mat()},
		{"Output", cv::Mat()},
	};

	// Operating data
	// Main window
	int active_image_index = 0;
	std::string active_stage = "Source";
	bool save_image = false;
	bool save_subimage = false;
	cv::Mat cards;
	cv::Mat display_image;

	// Sub window
	int active_card_index = 0;
	int active_substage_index = 0;
	std::string active_substage = "Warped";
	cv::Mat sub_display_image; 

	// Configure web cam parameters 
	cv::Mat cam_frame; 
	cv::VideoCapture camera(0);
	bool camera_available = true;
	bool use_camera = false;
	if (!camera.isOpened())
	{
		std::cout << "Cannot connect to camera... offline mode only\n";
		camera_available = false;
	}
	
	// Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
	cvui::init(WINDOW_NAME);
	auto time = std::chrono::high_resolution_clock::now();

	while (true) 
	{
		std::vector<std::unordered_map<std::string, cv::Mat>> card_data = {};

		// FPS Tracking 
		auto now = std::chrono::high_resolution_clock::now();
		auto time_delta = now - time;
		time = now;
		auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(time_delta).count() / 1000.0;
		std::cout << "Frame time (ms): " << frame_time << " | "
				  << "FPS(): " << 1.0 / (frame_time / 1000) << "\n";

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
			std::stringstream ss;
			ss << "base_image_stage_" << active_stage << ".png";
			cv::imwrite(ss.str(), pipe_out[active_stage]);
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
		int image_index = 0;
		for (auto& img: card_images)
		{
			// Initialize card data for viewer
			card_data.push_back(card_img_data);
			std::string best_match = "";

			auto& card_map = card_data[image_index];
			
			// Extract + Identify Rank
			cv::Rect rank_bounding_box(0, 0, 35, 55);
			cv::Mat rank_image = img(rank_bounding_box);
			
			// Draw bounding box on card
			cv::Mat card_img_color;
			cv::cvtColor(img, card_img_color, cv::COLOR_GRAY2BGR);
			cv::rectangle(card_img_color, rank_bounding_box, CV_RGB(0, 0, 255), 1);

			card_map["Rank"] = rank_image;
	
			cv::Mat rank_thresholded;
			cv::threshold(rank_image, rank_thresholded, 150, 255, cv::THRESH_OTSU);
			card_map["Rank Threshold"] = rank_thresholded.clone();
			rank_thresholded = ~rank_thresholded;

			cv::Mat rank_dilated;
			auto element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(4,4));
			cv::dilate(rank_thresholded, rank_dilated, element);
			card_map["Rank Dilated"] = ~rank_dilated;

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

			// Draw contours
			cv::Mat rank_contour_base; 
			cv::cvtColor(~rank_dilated, rank_contour_base, cv::COLOR_GRAY2BGR);

			if (!rank_contour_base.empty())
			{
				for (int i = 0; i < contours.size(); i++)
				{
					cv::drawContours(rank_contour_base, contours, i, { 255, 0, 0 }, 1);
				}
			}
			
			card_map["Rank Contours"] = rank_contour_base;

			cv::Mat bounded_rank = cv::Mat::zeros(rank_image.size(), CV_8UC1);
			if (!largest_c.empty())
			{
				bounded_rank = rank_dilated(bb);
			}
			card_map["Rank Bounded"] = ~bounded_rank;

			cv::Mat rank_identity = ~bounded_rank;
			card_map["Rank Final"] = rank_identity;

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
		
			cv::Rect suit_bounding_box(0, 55, 35, 45);
			cv::rectangle(card_img_color, suit_bounding_box, CV_RGB(0, 255, 0), 1);
			card_map["Warped"] = card_img_color;

			// Extract + Identify Suit 
			cv::Mat suit_image = img(suit_bounding_box);
			card_map["Suit"] = suit_image;
		
			cv::Mat suit_thresholded; 
			cv::threshold(suit_image, suit_thresholded, 120, 255, cv::THRESH_OTSU);
			card_map["Suit Threshold"] = suit_thresholded.clone();
			suit_thresholded = ~suit_thresholded;
		
			// Closing op for cutoff club stems
			cv::Mat suit_dilated;
			element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(1, 1));
			cv::dilate(suit_thresholded, suit_dilated, element);
			card_map["Suit Dilated"] = ~suit_dilated;

			cv::Mat suit_eroded;
			cv::erode(suit_dilated, suit_eroded, element);
			card_map["Suit Eroded"] = ~suit_eroded;
			
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

			// Draw suit contours
			cv::Mat suit_contour_img;
			cv::cvtColor(~suit_dilated, suit_contour_img, cv::COLOR_GRAY2BGR);

			if (!suit_contour_img.empty())
			{
				for (int i = 0; i < suit_contours.size(); i++)
				{
					cv::drawContours(suit_contour_img, suit_contours, i, { 255, 0, 0 }, 1);
				}
			}
			
			card_map["Suit Contours"] = suit_contour_img;

			cv::Mat bounded_suit = suit_dilated.clone();
			if (!suit_bb.empty())
			{
				bounded_suit = suit_dilated(suit_bb);
			}
			
			// Final suit 
			bounded_suit = ~bounded_suit;
			card_map["Suit Bounded"] = bounded_suit;

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
			image_index++;
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

		// Select active sub image stage
		active_substage = sub_stage_titles[active_substage_index];
		if (card_data.empty())
		{
			sub_display_image = cv::Mat::zeros(sub_image.heightWithoutBorders(), sub_image.widthWithoutBorders(), CV_8UC3);
		}
		else
		{
			active_card_index = active_card_index > card_data.size() - 1 ? card_data.size() - 1 : active_card_index;
			sub_display_image = card_data[active_card_index][active_substage];
		}

		if (save_subimage)
		{
			std::stringstream ss;
			ss << "card_" << active_card_index << "_stage_" << active_substage << ".png";
			cv::imwrite(ss.str(), sub_display_image);
			save_subimage = false;
		}
		
		// Render the settings window and its content, if it is not minimized.
		settings.begin(frame);
		if (!settings.isMinimized()) 
		{
			int width = 300;
			cvui::text("Pipeline Stage - " + active_stage);
			cvui::trackbar(width, &active_image_index, 0, (int)stage_titles.size() - 1, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);

			// Hanging 
			if (card_data.size() > 1)
			{
				cvui::text("Active Card - " + std::to_string(active_card_index));
				cvui::trackbar(width, &active_card_index, 0, (int)card_data.size() - 1, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
				cvui::space(10);
			}
			
			cvui::text("Suit/Rank Stage - " + active_substage);
			cvui::trackbar(width, &active_substage_index, 0, (int)sub_stage_titles.size() - 1, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);

			cvui::text("Save Active Image");
			cvui::space(8);
			save_image = cvui::button("Save");
			cvui::space(10);

			cvui::text("Save Card View");
			cvui::space(8);
			save_subimage = cvui::button("Save");
			cvui::space(10);

			if (camera_available)
			{
				cvui::text("Save Current Image");
				cvui::space(8);
				static bool old = use_camera;
				static bool current = old;
				current = cvui::checkbox("Live", &use_camera);
				if (current != old)
				{
					active_card_index = 0;
					old = current;;
				}
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
			// Check image type 

			cvui::image(disp);
		}
		image.end();

		sub_image.begin(frame);
		if (!sub_image.isMinimized())
		{
			cv::Mat disp; 
			if (card_data.empty())
			{
				disp = sub_display_image;
			}
			else if (active_substage == "Suit Contours" || active_substage == "Rank Contours" || active_substage == "Warped")
			{
				disp = sub_display_image;
			}
			else
			{
				cvtColor(sub_display_image, disp, cv::COLOR_GRAY2BGR);
			}
			cvui::image(disp);
		}
		sub_image.end();

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


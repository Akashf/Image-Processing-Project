#include "DeckTemplate.h"
#include "PipelineStages.h"
#include "Pipeline.h"
#include "Utilities.h"
#include "CardDetector.h"

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


int main() 
{   
	// OpenCV config
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

	// CVUI Window config
	const std::string windowName = "Most Constrained Card Detector";
	const size_t window_height = 1080;
	const size_t window_width = 1920;
	cv::Mat frame = cv::Mat(window_height, window_width, CV_8UC3);

	// Source image 
	const std::string test_image_root = "assets/test_images/";
	cv::Mat source = cv::imread(test_image_root + "cards-numerous.jpg");
	cv::Mat cards_color = source;

	// Load deck template
	mccd::DeckTemplateParams templateParams; 
	templateParams.folder = "assets/template_images/";
	templateParams.rankSize = { 30, 45 };
	templateParams.ext = "png";

	// Gaussian Parameters 
	mccd::GaussianParameters gauss_params; 
	gauss_params.kernel_size = 3;
	gauss_params.sigma = 0;

	// Canny parameters 
	mccd::CannyParameters canny_params; 
	canny_params.low_threshold = 0;
	canny_params.high_threshold = 255;

	// Contour parameters (defaults)
	mccd::ContourParameters contour_params;

	// Card detector
	mccd::CardDetector cardDetector(templateParams);

	// Create windows
	EnhancedWindow settings(0, 0, 320, window_height, "Settings");
	EnhancedWindow image(settings.width(), 0, cards_color.cols + 20, cards_color.rows + 40, "Active Image");
	EnhancedWindow sub_image(settings.width(), window_height - 500, 350, 500, "Individual Card View");

	// Operating data - main window
	size_t active_image_index = 0;
	std::string active_stage = "Source";
	bool save_image = false;
	bool save_subimage = false;
	cv::Mat display_image;

	// Operating data - Sub window
	size_t active_card_index = 0;
	size_t active_substage_index = 0;
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
	
	// Init cvui and tell it to create a OpenCV window
	cvui::init(windowName);
	auto time = std::chrono::high_resolution_clock::now();

	while (true) 
	{
		// FPS Tracking 
		auto now = std::chrono::high_resolution_clock::now();
		auto time_delta = now - time;
		time = now;
		auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(time_delta).count() / 1000.0;
		std::cout 
			<< "Frame time (ms): " << frame_time << " | "
			<< "FPS(): " << 1.0 / (frame_time / 1000) << "\n";

		// Clear frame 
		frame = cv::Scalar(53, 101, 77);

		// Read image from camera if present 
		if (use_camera)
		{
			camera >> cam_frame;
			cards_color = cam_frame;
		}
		else
		{
			cards_color = source.clone();
		}

		// Execute card feature detection and identification 
		cardDetector.update
		(
			cards_color, 
			gauss_params,
			canny_params,
			contour_params
		);

		const auto& pipelineOut = cardDetector.getPipelineOutput();
		const auto& cardData = cardDetector.getCardData();
 
		// Resize image window to fit camera frame 
		int newHeight = cards_color.rows + 40;
		int newWidth = cards_color.cols + 20;

		image.setHeight(newHeight);
		image.setWidth(newWidth);

		// Select active stage 
		active_stage = mccd::stage_titles[active_image_index];
		display_image = pipelineOut.at(active_stage);

		// Select active sub image 
		active_substage = mccd::sub_stage_titles[active_substage_index];

		if (cardData.empty())
		{
			sub_display_image = cv::Mat::zeros(sub_image.heightWithoutBorders(), sub_image.widthWithoutBorders(), CV_8UC3);
		}
		else
		{
			active_card_index = active_card_index > cardData.size() - 1 ? cardData.size() - 1 : active_card_index;
			sub_display_image = cardData.at(active_card_index).at(active_substage);
		}

		if (save_image)
		{
			std::stringstream ss;
			ss << "base_image_stage_" << active_stage << ".png";
			cv::imwrite(ss.str(), display_image);
			save_image = false;
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
			const size_t sliderWidth = 300;
			cvui::text("Pipeline Stage - " + active_stage);
			cvui::trackbar<size_t>(sliderWidth, &active_image_index, 0, mccd::stage_titles.size() - 1, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);

			// Hanging 
			if (cardData.size() > 1)
			{
				cvui::text("Active Card - " + std::to_string(active_card_index));
				cvui::trackbar<size_t>(sliderWidth, &active_card_index, 0, cardData.size() - 1, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
				cvui::space(10);
			}
			
			cvui::text("Suit/Rank Stage - " + active_substage);
			cvui::trackbar<size_t>(sliderWidth, &active_substage_index, 0, mccd::sub_stage_titles.size() - 1, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
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
					old = current;
				}
				cvui::space(10);
			}
			
			cvui::text("Gaussian Configuration");
			cvui::space(10);
			
			cvui::text("Kernel Size");
			cvui::space(10);
			cvui::trackbar<size_t>(sliderWidth, &gauss_params.kernel_size, 1, 9, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 2);

			cvui::text("Sigma");
			cvui::space(10);
			cvui::trackbar<float>(sliderWidth, &gauss_params.sigma, 0, 10, 0.1, "%0.1f");
			cvui::space(10);

			cvui::text("Canny Configuration");
			cvui::space(10);
			cvui::text("Low Threshold");
			cvui::space(4);
			cvui::trackbar<size_t>(sliderWidth, &canny_params.low_threshold, 0, 255, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(4);
			cvui::text("High Threshold");
			cvui::space(4);
			cvui::trackbar<size_t>(sliderWidth, &canny_params.high_threshold, 0, 255, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
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

		sub_image.begin(frame);
		if (!sub_image.isMinimized())
		{
			cv::Mat disp; 
			if (cardData.empty())
			{
				disp = sub_display_image;
			}
			else if (
				active_substage == "Suit Contours" 
				|| active_substage == "Rank Contours" 
				|| active_substage == "Warped"
			)
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

		// Update cvui and draw frame
		cvui::imshow(windowName, frame);

		// Check if esc or 'X' was pressed
		const bool escPressed = cv::waitKey(30) == 27;
		const bool exitPressed = cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1;
		if (escPressed || exitPressed) 
		{
			break;
		}
	}

	return 0;
}

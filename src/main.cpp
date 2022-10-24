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
	const size_t windowHeight = 1080;
	const size_t windowWidth = 1920;
	cv::Mat frame = cv::Mat(windowHeight, windowWidth, CV_8UC3);
	// TODO: Background clear color 
	// TODO: Window clear color 

	// Source image 
	const std::string testImageFolder = "assets/test_images/";
	cv::Mat source = cv::imread(testImageFolder + "cards-numerous.jpg");
	cv::Mat cardsColor = source;

	// Load deck template
	mccd::DeckTemplateParams templateParams;
	templateParams.folder = "assets/template_images/";
	templateParams.rankSize = { 30, 45 };
	templateParams.ext = "png";

	// Gaussian Parameters 
	mccd::GaussianParameters gaussParams; 
	gaussParams.kernelSize = 3;
	gaussParams.sigma = 0;

	// Canny parameters 
	mccd::CannyParameters cannyParams; 
	cannyParams.lowThreshold = 0;
	cannyParams.highThreshold = 255;

	// Contour parameters (defaults)
	mccd::ContourParameters contourParams;

	// Card detector
	mccd::CardDetector cardDetector(templateParams);

	// Create windows
	EnhancedWindow settingsWindow(0, 0, 320, windowHeight, "Settings");
	EnhancedWindow mainImageWindow(settingsWindow.width(), 0, cardsColor.cols + 20, cardsColor.rows + 40, "Active Image");
	EnhancedWindow subImageWindow(settingsWindow.width(), windowHeight - 500, 350, 500, "Individual Card View");

	// Operating data - main window
	size_t activeImageIndex = 0;
	std::string activeStageName = "Source";
	bool shouldSaveImage = false;
	bool shouldSaveSubimage = false;
	cv::Mat displayImage;

	// Operating data - Sub window
	size_t activeCardIndex = 0;
	size_t activeSubstageIndex = 0;
	std::string activeSubstageName = "Warped";
	cv::Mat subDisplayImage; 

	// Configure web cam parameters 
	cv::Mat camFrame; 
	cv::VideoCapture camera(0);
	bool cameraAvailable = true;
	bool useCamera = false;
	if (!camera.isOpened())
	{
		std::cout << "Cannot connect to camera... offline mode only\n";
		cameraAvailable = false;
	}
	
	// Init cvui and tell it to create a OpenCV window
	cvui::init(windowName);
	auto time = std::chrono::high_resolution_clock::now();

	while (true) 
	{
		// FPS Tracking 
		const auto now = std::chrono::high_resolution_clock::now();
		const auto timeDelta = now - time;
		const auto frameTimeMs = std::chrono::duration_cast<std::chrono::microseconds>(timeDelta).count() / 1000.0;
		time = now;
		std::cout 
			<< "Frame time (ms): " << frameTimeMs << " | "
			<< "FPS(): " << 1.0 / (frameTimeMs / 1000) << "\n";

		// Clear frame 
		frame = cv::Scalar(53, 101, 77);

		// Read image from camera if present 
		if (useCamera && cameraAvailable)
		{
			camera >> camFrame;
			cardsColor = camFrame;
		}
		else
		{
			cardsColor = source.clone();
		}

		// Execute card feature detection and identification 
		cardDetector.update
		(
			cardsColor, 
			gaussParams,
			cannyParams,
			contourParams
		);

		const auto& pipelineOut = cardDetector.getPipelineOutput();
		const auto& cardData = cardDetector.getCardData();
 
		// Resize image window to fit camera frame 
		int newHeight = cardsColor.rows + 40;
		int newWidth = cardsColor.cols + 20;

		mainImageWindow.setHeight(newHeight);
		mainImageWindow.setWidth(newWidth);

		// Select active stage 
		activeStageName = mccd::stageTitles[activeImageIndex];
		displayImage = pipelineOut.at(activeStageName);

		// Select active sub image 
		activeSubstageName = mccd::subStageTitles[activeSubstageIndex];

		if (cardData.empty())
		{
			subDisplayImage = cv::Mat::zeros(subImageWindow.heightWithoutBorders(), subImageWindow.widthWithoutBorders(), CV_8UC3);
		}
		else
		{
			activeCardIndex = activeCardIndex > cardData.size() - 1 ? cardData.size() - 1 : activeCardIndex;
			subDisplayImage = cardData.at(activeCardIndex).at(activeSubstageName);
		}

		if (shouldSaveImage)
		{
			std::stringstream ss;
			ss << "base_image_stage_" << activeStageName << ".png";
			cv::imwrite(ss.str(), displayImage);
			shouldSaveImage = false;
		}

		if (shouldSaveSubimage)
		{
			std::stringstream ss;
			ss << "card_" << activeCardIndex << "_stage_" << activeSubstageName << ".png";
			cv::imwrite(ss.str(), subDisplayImage);
			shouldSaveSubimage = false;
		}
		
		// Render the settings window and its content, if it is not minimized.
		settingsWindow.begin(frame);
		if (!settingsWindow.isMinimized()) 
		{
			const size_t sliderWidth = 300;
			cvui::text("Pipeline Stage - " + activeStageName);
			cvui::trackbar<size_t>(sliderWidth, &activeImageIndex, 0, mccd::stageTitles.size() - 1, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);

			// Hanging 
			if (cardData.size() > 1)
			{
				cvui::text("Active Card - " + std::to_string(activeCardIndex));
				cvui::trackbar<size_t>(sliderWidth, &activeCardIndex, 0, cardData.size() - 1, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
				cvui::space(10);
			}
			
			cvui::text("Suit/Rank Stage - " + activeSubstageName);
			cvui::trackbar<size_t>(sliderWidth, &activeSubstageIndex, 0, mccd::subStageTitles.size() - 1, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);

			cvui::text("Save Active Image");
			cvui::space(8);
			shouldSaveImage = cvui::button("Save");
			cvui::space(10);

			cvui::text("Save Card View");
			cvui::space(8);
			shouldSaveSubimage = cvui::button("Save");
			cvui::space(10);

			if (cameraAvailable)
			{
				cvui::text("Save Current Image");
				cvui::space(8);
				static bool old = useCamera;
				static bool current = old;
				current = cvui::checkbox("Live", &useCamera);
				if (current != old)
				{
					activeCardIndex = 0;
					old = current;
				}
				cvui::space(10);
			}
			
			cvui::text("Gaussian Configuration");
			cvui::space(10);
			
			cvui::text("Kernel Size");
			cvui::space(10);
			cvui::trackbar<size_t>(sliderWidth, &gaussParams.kernelSize, 1, 9, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 2);

			cvui::text("Sigma");
			cvui::space(10);
			cvui::trackbar<float>(sliderWidth, &gaussParams.sigma, 0, 10, 0.1, "%0.1f");
			cvui::space(10);

			cvui::text("Canny Configuration");
			cvui::space(10);
			cvui::text("Low Threshold");
			cvui::space(4);
			cvui::trackbar<size_t>(sliderWidth, &cannyParams.lowThreshold, 0, 255, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(4);
			cvui::text("High Threshold");
			cvui::space(4);
			cvui::trackbar<size_t>(sliderWidth, &cannyParams.highThreshold, 0, 255, 1, "%0.1f", cvui::TRACKBAR_DISCRETE, 1);
			cvui::space(10);
		}
		settingsWindow.end();
		
		mainImageWindow.begin(frame);
		if (!mainImageWindow.isMinimized())
		{
			cv::Mat disp; 
			if (activeStageName == "Contours" || activeStageName == "Output" || activeStageName == "Rectangle Contours")
			{
				disp = displayImage;
			}
			else
			{
				cv::cvtColor(displayImage, disp, cv::COLOR_GRAY2BGR);
			}
			cvui::image(disp);
		}
		mainImageWindow.end();

		subImageWindow.begin(frame);
		if (!subImageWindow.isMinimized())
		{
			cv::Mat disp; 
			if (cardData.empty())
			{
				disp = subDisplayImage;
			}
			else if (
				activeSubstageName == "Suit Contours" 
				|| activeSubstageName == "Rank Contours" 
				|| activeSubstageName == "Warped"
			)
			{
				disp = subDisplayImage;
			}
			else
			{
				cvtColor(subDisplayImage, disp, cv::COLOR_GRAY2BGR);
			}
			cvui::image(disp);
		}
		subImageWindow.end();

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

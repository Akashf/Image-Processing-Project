#pragma once 

#include "DeckTemplate.h"
#include "Pipeline.h"

#include <string>
#include <vector>

#include "opencv2/core.hpp"


namespace mccd {

    const std::vector<std::string> sub_stage_titles = 
	{
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

	const std::vector<std::string> stage_titles = 
	{
		"Source",
		"Blurred",
		"Equalized",
		"Edges",
		"Contours",
		"Rectangle Contours",
		"Output"
	};

    // Just card detection data, nothing UI related 
    class CardDetector
    {
    public: 
        CardDetector(DeckTemplateParams deckTemplateParams);

        void update
        (
			cv::Mat inputImage,
			GaussianParameters gaussParams,
			CannyParameters cannyParams,
			ContourParameters contourParams
		);

		const std::vector<cv::Mat>& getCardImages() const { return m_cardImages; }
		const std::vector<std::unordered_map<std::string, cv::Mat>>& getCardData() const { return m_cardData; }
		const std::unordered_map<std::string, cv::Mat>& getPipelineOutput() const { return m_cardPipeOuts; }

    private:
		void extractCardImagesFromContours();
        void extractAndIdentifyRankAndSuit();
		void generateContourOverlay();
		void generateRectangularContourOverlay();
		void generateGuessAnnotatedOverlay();

		DeckTemplate m_deckTemplate;

		// -- Operating data

        // Sub stage outputs for each card
		cv::Mat m_activeImageColor = {};
		cv::Mat m_activeImageGrey = {};

        std::vector<std::unordered_map<std::string, cv::Mat>> m_cardData = {};
		std::vector<cv::Mat> m_cardImages = {};
		std::vector<cv::Point> m_cardMidpoints = {};
		std::vector<cv::Rect> m_boundingRects = {};
		mccd::Contours m_rectContours = {};
		mccd::Contours m_contours = {};
		std::vector<std::string> m_cardBestGuesses = {};
		std::vector<std::string> m_suitBestGuesses = {};
		const std::vector<cv::Point2f> m_targetPts= { {0, 0}, {0, 349}, {249, 349}, {249, 0} };

        // -- Pipeline outputs
		std::unordered_map<std::string, cv::Mat> m_cardPipeOuts =
		{
			{"Source", cv::Mat::zeros(10, 10, CV_8UC1)},
			{"Blurred", cv::Mat()},
			{"Equalized", cv::Mat()},
			{"Edges", cv::Mat()},
			{"Contours", cv::Mat()},
			{"Rectangle Contours", cv::Mat()},
			{"Output", cv::Mat()},
		};

        std::unordered_map<std::string, cv::Mat> m_cardImgDataDefault =
        {
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
    };
	
}; // namespace mccd

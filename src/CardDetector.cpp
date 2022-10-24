#include "CardDetector.h"

namespace mccd {

    CardDetector::CardDetector(DeckTemplateParams deckTemplateParams) 
		: m_deckTemplate(DeckTemplate(deckTemplateParams))
    {
    }

    void CardDetector::update(
        cv::Mat inputImage,
        GaussianParameters gaussParams,
        CannyParameters cannyParams,
        ContourParameters contourParams
    )
    {
		m_activeImageColor = inputImage;
		cv::cvtColor(m_activeImageColor, m_activeImageGrey, cv::COLOR_BGR2GRAY);

        // Clear data
        m_cardImages.clear();
        m_cardData.clear();
        m_cardMidpoints.clear();
        m_rectContours.clear();
        m_cardBestGuesses.clear();
        m_suitBestGuesses.clear();
        m_boundingRects.clear();

        // Create and run pipeline 
        mccd::CardExtractionPipeline pipeline
        (
			m_activeImageGrey,
            gaussParams, 
            cannyParams,
            contourParams
        );

		// TODO: Update cardPipeOut with images from the pipeline output
        pipeline.execute();
		auto& output = pipeline.getStageOutputs();
		m_cardPipeOuts["Source"] = output["Source"].clone();
		m_cardPipeOuts["Blurred"] = output["Blurred"].clone();
		m_cardPipeOuts["Equalized"] = output["Equalized"].clone();
		m_cardPipeOuts["Edges"] = output["Edges"].clone();

        m_contours = pipeline.getOutput();
        extractCardImagesFromContours();
		extractAndIdentifyRankAndSuit();
		generateContourOverlay();
		generateRectangularContourOverlay();
		generateGuessAnnotatedOverlay();
    }

    void CardDetector::extractCardImagesFromContours() 
    {
        size_t i = 0;
        for (auto& c : m_contours)
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
            for (auto p : output)
            {
                output_i.push_back(cv::Point((int)p.x, (int)p.y));

                x_sum += p.x;
                y_sum += p.y;
            }
            cv::Point2f mid(x_sum / 4, y_sum / 4);
            m_cardMidpoints.push_back(mid);
            m_rectContours.push_back(output_i);

            // Determine semantic location of this point in image
            std::vector<cv::Point2f> src(4);
            for (auto p : output)
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

            cv::Mat p = cv::getPerspectiveTransform(src, m_targetPts);
            cv::Mat img;

            cv::warpPerspective(m_activeImageGrey, img, p, cv::Size(250, 350));
            m_cardImages.push_back(img);
        }
    }

    void CardDetector::extractAndIdentifyRankAndSuit()
    {
        // Extract + identify rank / suit
		size_t i = 0;
		size_t imageIndex = 0;
		for (auto& img: m_cardImages)
		{
			// Initialize card data 
			m_cardData.push_back(m_cardImgDataDefault);
			std::string bestMatch = "";

			auto& card_map = m_cardData[imageIndex];
			
			// Extract + Identify Rank
			cv::Rect rankBoundingBox(0, 0, 35, 55);
			cv::Mat rankImage = img(rankBoundingBox);
			
			// Draw bounding box on card
			cv::Mat cardImgColor;
			cv::cvtColor(img, cardImgColor, cv::COLOR_GRAY2BGR);
			cv::rectangle(cardImgColor, rankBoundingBox, CV_RGB(0, 0, 255), 1);

			card_map["Rank"] = rankImage;
	
			cv::Mat rankThresholded;
			cv::threshold(rankImage, rankThresholded, 150, 255, cv::THRESH_OTSU);
			card_map["Rank Threshold"] = rankThresholded.clone();
			rankThresholded = ~rankThresholded;

			cv::Mat rankDilated;
			auto element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(4,4));
			cv::dilate(rankThresholded, rankDilated, element);
			card_map["Rank Dilated"] = ~rankDilated;

			std::vector<std::vector<cv::Point>> contours; 
			cv::findContours(rankDilated, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			// Select largest contour
			mccd::Contour largestCountour;
			float maxArea = 0;
			for (auto& c: contours)
			{
				float area = cv::contourArea(c);
				if (area > maxArea)
				{
					maxArea = area;
					largestCountour = c;
				}
			}

			// Draw bounding box of largest contour
			cv::Rect bb = cv::boundingRect(largestCountour);

			// Draw contours
			cv::Mat rankCountourBase; 
			cv::cvtColor(~rankDilated, rankCountourBase, cv::COLOR_GRAY2BGR);

			if (!rankCountourBase.empty())
			{
				for (int i = 0; i < contours.size(); i++)
				{
					cv::drawContours(rankCountourBase, contours, i, { 255, 0, 0 }, 1);
				}
			}
			
			card_map["Rank Contours"] = rankCountourBase;

			cv::Mat boundedRank = cv::Mat::zeros(rankImage.size(), CV_8UC1);
			if (!largestCountour.empty())
			{
				boundedRank = rankDilated(bb);
			}
			card_map["Rank Bounded"] = ~boundedRank;

			cv::Mat rankIdentity = ~boundedRank;
			card_map["Rank Final"] = rankIdentity;

			int minDiff = std::numeric_limits<int>().max();
			float maxConfidence = 0;
			for (const auto& img: m_deckTemplate.getRankTemplateImages())
			{
				cv::Mat diffImage; 
				cv::Mat templateImage;
				cv::resize(img.second, templateImage, rankIdentity.size());
				cv::absdiff(rankIdentity, templateImage, diffImage);
				int avgDiff = cv::sum(diffImage)[0] / 255;
				if (avgDiff < minDiff)
				{
					minDiff = avgDiff;
					bestMatch = img.first;
				}	
			}

			m_cardBestGuesses.push_back(bestMatch);
		
			cv::Rect suitBoundingBox(0, 55, 35, 45);
			cv::rectangle(cardImgColor, suitBoundingBox, CV_RGB(0, 255, 0), 1);
			card_map["Warped"] = cardImgColor;

			// Extract + Identify Suit 
			cv::Mat suitImage = img(suitBoundingBox);
			card_map["Suit"] = suitImage;
		
			cv::Mat suitThresholded; 
			cv::threshold(suitImage, suitThresholded, 120, 255, cv::THRESH_OTSU);
			card_map["Suit Threshold"] = suitThresholded.clone();
			suitThresholded = ~suitThresholded;
		
			// Closing op for cutoff club stems
			cv::Mat suitDilated;
			element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(1, 1));
			cv::dilate(suitThresholded, suitDilated, element);
			card_map["Suit Dilated"] = ~suitDilated;

			cv::Mat suitEroded;
			cv::erode(suitDilated, suitEroded, element);
			card_map["Suit Eroded"] = ~suitEroded;
			
			std::vector<std::vector<cv::Point>> suitContours;
			cv::findContours(suitDilated, suitContours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			// Calculate bb of largest area contour
			cv::Rect suit_bb; 
			{
				mccd::Contour largestContour;

				float maxArea = 0;
				for (auto& c : suitContours)
				{
					float area = cv::contourArea(c);
					if (area > maxArea)
					{
						maxArea = area;
						largestContour = c;
					}
				}
				suit_bb = cv::boundingRect(largestContour);
			}

			// Draw suit contours
			cv::Mat suitContourImage;
			cv::cvtColor(~suitDilated, suitContourImage, cv::COLOR_GRAY2BGR);

			if (!suitContourImage.empty())
			{
				for (int i = 0; i < suitContours.size(); i++)
				{
					cv::drawContours(suitContourImage, suitContours, i, { 255, 0, 0 }, 1);
				}
			}
			
			card_map["Suit Contours"] = suitContourImage;

			cv::Mat boundedSuit = suitDilated.clone();
			if (!suit_bb.empty())
			{
				boundedSuit = suitDilated(suit_bb);
			}
			
			// Final suit 
			boundedSuit = ~boundedSuit;
			card_map["Suit Bounded"] = boundedSuit;

			// Identify rank 
			minDiff = std::numeric_limits<int>().max();
			std::string suitMatch = "";
			for (const auto& img : m_deckTemplate.getSuitTemplateImages())
			{
				cv::Mat diffImage;
				cv::Mat templateImage;
				cv::resize(img.second, templateImage, boundedSuit.size());
				cv::absdiff(boundedSuit, templateImage, diffImage);
				int avgDiff = cv::sum(diffImage)[0] / 255;
				if (avgDiff < minDiff)
				{
					minDiff = avgDiff;
					suitMatch = img.first;
				}
			}

			m_suitBestGuesses.push_back(suitMatch);
			imageIndex++;
		}
    }

	void CardDetector::generateContourOverlay()
	{
		// Generate original contour overlay
		cv::Mat contourBase = m_activeImageColor.clone();
		for (size_t i = 0; i < m_contours.size(); i++)
		{
			cv::drawContours(contourBase, m_contours, i, cv::Scalar(0, 0, 255), 4);
		}
		m_cardPipeOuts["Contours"] = contourBase.clone();
	}

	void CardDetector::generateRectangularContourOverlay()
	{
		// Generate rectangle contour overlay
		cv::Mat rectContourBase = m_activeImageColor.clone();
		for (size_t i = 0; i < m_rectContours.size(); i++)
		{
			cv::drawContours(rectContourBase, m_rectContours, i, cv::Scalar(0, 0, 255), 2);
		}
		m_cardPipeOuts["Rectangle Contours"] = rectContourBase.clone();
	}

	void CardDetector::generateGuessAnnotatedOverlay()
	{
		cv::Mat outputBase = m_activeImageColor.clone();
		for (size_t i = 0; i < m_rectContours.size(); i++)
		{
			cv::drawContours(outputBase, m_rectContours, i, cv::Scalar(0, 0, 255), 2);
		}

		// Draw best match rank and suit at center of image 
		for (size_t i = 0; i < m_cardImages.size(); i++)
		{
			auto mid = m_cardMidpoints[i];
			std::string rankBestGuess = m_cardBestGuesses[i];
			std::string suitBestGuess = m_suitBestGuesses[i];

			cv::Size rankSize = cv::getTextSize(rankBestGuess, cv::FONT_HERSHEY_COMPLEX, 1, 2, nullptr);
			cv::Point rankOrigin = cv::Point(mid.x - rankSize.width / 2, mid.y + rankSize.height / 2);

			cv::Size suitSize = cv::getTextSize(suitBestGuess, cv::FONT_HERSHEY_COMPLEX, 0.75, 2, nullptr);
			cv::Point sutiOrigin = cv::Point(mid.x - suitSize.width / 2, mid.y + suitSize.height / 2);

			cv::putText(outputBase, rankBestGuess, rankOrigin, cv::FONT_HERSHEY_COMPLEX, 1.0, CV_RGB(0, 0, 255), 2);
			cv::putText(outputBase, suitBestGuess, sutiOrigin + cv::Point(0, 24), cv::FONT_HERSHEY_COMPLEX, 0.75, CV_RGB(0, 0, 255), 2);
		}
		m_cardPipeOuts["Output"] = outputBase.clone();
	}
	
}; // namespace mccd

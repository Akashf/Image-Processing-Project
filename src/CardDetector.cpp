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
		int image_index = 0;
		for (auto& img: m_cardImages)
		{
			// Initialize card data 
			m_cardData.push_back(m_cardImgDataDefault);
			std::string best_match = "";

			auto& card_map = m_cardData[image_index];
			
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
			for (const auto& img: m_deckTemplate.getRankTemplateImages())
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

			m_cardBestGuesses.push_back(best_match);
		
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
			for (const auto& img : m_deckTemplate.getSuitTemplateImages())
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

			m_suitBestGuesses.push_back(suit_match);
			image_index++;
		}
    }

	void CardDetector::generateContourOverlay()
	{
		// Generate original contour overlay
		cv::Mat contour_base = m_activeImageColor.clone();
		for (size_t i = 0; i < m_contours.size(); i++)
		{
			cv::drawContours(contour_base, m_contours, i, cv::Scalar(0, 0, 255), 4);
		}
		m_cardPipeOuts["Contours"] = contour_base.clone();
	}

	void CardDetector::generateRectangularContourOverlay()
	{
		// Generate rectangle contour overlay
		cv::Mat rect_contour_base = m_activeImageColor.clone();
		for (size_t i = 0; i < m_rectContours.size(); i++)
		{
			cv::drawContours(rect_contour_base, m_rectContours, i, cv::Scalar(0, 0, 255), 2);
		}
		m_cardPipeOuts["Rectangle Contours"] = rect_contour_base.clone();
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

#include "DeckTemplate.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"


DeckTemplate::DeckTemplate(cv::Size rankSize, std::string folder, std::string extension)
    : m_rankSize(std::move(rankSize))
    , m_folder(std::move(folder))
    , m_extension(std::move(extension))
{
    loadTemplateImages();
}

void DeckTemplate::loadTemplateImages()
{
    loadRankTemplateImages();
    loadSuitTemplateImages();
}

void DeckTemplate::loadRankTemplateImages()
{
    size_t i = 0;
    for (const auto& rank: m_rankNames)
    {
        cv::Mat img = cv::imread(m_folder + rank + "." + m_extension, cv::IMREAD_GRAYSCALE);
        cv::Mat resized;
        cv::resize(img, resized, m_rankSize);
        m_rankTemplateImages[i] = {rank, resized};
        i++; 
    }
}

void DeckTemplate::loadSuitTemplateImages()
{
    size_t i = 0;
    for (const auto& suit: m_suitNames)
    {
        m_suitTemplateImages[i] = 
        {
            suit, 
            cv::imread(m_folder + suit + "." + m_extension, cv::IMREAD_GRAYSCALE) 
        };   
        i++;
    }
}
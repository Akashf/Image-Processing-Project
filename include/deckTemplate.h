#pragma once

#include <string>
#include <vector>

#include "opencv2/core.hpp"


namespace mccd {

    using NamedImage = std::pair<std::string, cv::Mat>;
    struct DeckTemplateParams
    {
        std::string folder;
        std::string ext;
        cv::Size_<size_t> rankSize;
    };

    class DeckTemplate
    {
    public:
        DeckTemplate(cv::Size rankSize, std::string folder, std::string extension);
        DeckTemplate(DeckTemplateParams params);
        const std::array<NamedImage, 13>& getRankTemplateImages() { return m_rankTemplateImages; }
        const std::array<NamedImage, 4>& getSuitTemplateImages() { return m_suitTemplateImages; }

    private:
        void loadTemplateImages();
        void loadRankTemplateImages();
        void loadSuitTemplateImages();

        cv::Size m_rankSize = { 30, 45 };
        std::string m_folder = "";
        std::string m_extension = "";

        std::array<NamedImage, 13> m_rankTemplateImages = {};
        std::array<NamedImage, 4> m_suitTemplateImages = {};

        const std::array<std::string, 13> m_rankNames =
        {
            "Ace", "Two", "Three", "Four", "Five", "Six", "Seven",
            "Eight", "Nine", "Ten", "Jack", "Queen", "King"
        };

        const std::array<std::string, 4> m_suitNames =
        {
            "Hearts", "Clubs", "Spades", "Diamonds"
        };
    };

} // namespace mccd

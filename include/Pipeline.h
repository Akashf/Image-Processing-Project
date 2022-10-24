#pragma once

#include "PipelineStages.h"

#include <unordered_map>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"


namespace mccd {

    template <typename TIn, typename TOut>
    class GPipeline
    {
    public:
        GPipeline(std::string name, TIn in) 
            : m_in(std::move(in))
            , m_name(name) 
        {}
        virtual ~GPipeline() = default;

        virtual void execute() {}

        virtual void setInput(TIn in) { m_in = in; }
        virtual TOut getOutput() { return m_out; }

    protected: 
        TIn m_in; 
        TOut m_out;
        std::string m_name = "";
        sPtr<cv::GComputation> m_pipelineInternal = nullptr;
    };

    class CardExtractionPipeline : public GPipeline<cv::Mat, mccd::Contours> 
    {
    public: 
        CardExtractionPipeline(
            cv::Mat in,
            GaussianParameters gaussParams,
            CannyParameters cannyParams,
            ContourParameters contourParams
        )
            : GPipeline("CardExtraction", in)
            , m_gaussParams(gaussParams)
            , m_cannyParams(cannyParams)
            , m_contourParams(contourParams)
        {
            cv::GMat g_in = {};
            mccd::GGaussianBlur gaussianStage(g_in, m_gaussParams);
            mccd::GHistEqualize equalizeStage(gaussianStage.getOutput());
            mccd::GCannyEdges cannyStage(gaussianStage.getOutput(), m_cannyParams);
            mccd::GContours contourStage(cannyStage.getOutput(), m_contourParams);

            cv::GMat g_equalized = equalizeStage.getOutput();
            cv::GMat g_blurred = gaussianStage.getOutput();
            cv::GMat g_edges = cannyStage.getOutput();
            cv::GArray<mccd::GContour> g_contours = contourStage.getOutput();

            m_pipelineInternal = std::make_shared<cv::GComputation>
            (
                cv::GIn(g_in),
                cv::GOut(g_blurred, g_equalized, g_edges, g_contours)
            );
        }

        virtual void execute() override
        {
            // Execute pipeline
            m_stageOutputImages["Source"] = m_in;
            m_pipelineInternal->apply
            (
                cv::gin(m_in),
                cv::gout
                (
                    m_stageOutputImages["Blurred"],
                    m_stageOutputImages["Equalized"],
                    m_stageOutputImages["Edges"],
                    m_contours
                )
            );

            // Final output are card contours
            m_out = m_contours;
        }

        std::unordered_map<std::string, cv::Mat>& getStageOutputs()
        {
            return m_stageOutputImages;
        }

        mccd::Contours& getContours()
        {
            return m_contours;
        }
           
    protected:
        GaussianParameters m_gaussParams = {};
        CannyParameters m_cannyParams = {};
        ContourParameters m_contourParams = {};

        std::unordered_map<std::string, cv::Mat> m_stageOutputImages = {};
        mccd::Contours m_contours = {};
    };

} // namespace mccd

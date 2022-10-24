#pragma once 

#include "Utilities.h"

#include <string> 
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"

namespace mccd {

    using GContour = cv::GArray<cv::Point>;

    struct CannyParameters
    {
        size_t low_threshold = 0;
        size_t high_threshold = 255;
    };

    struct GaussianParameters
    {
        size_t kernel_size = 3;
        float sigma = 0;
    };

    struct ContourParameters
    {
        cv::RetrievalModes retMode = cv::RETR_EXTERNAL;
        cv::ContourApproximationModes approxMode = cv::CHAIN_APPROX_SIMPLE;
    };

    template<typename T>
    class GPipelineStage
    {
    public:
        GPipelineStage(std::string name, cv::GMat in)
            : m_in(std::move(in))
            , m_name(std::move(name))
        {
        }

        T getOutput() { return m_out; }
        std::string getName() const { return m_name; }

    protected:
        virtual void setOutput() = 0;

        cv::GMat m_in;
        T m_out;
        std::string m_name = "";
    };

    class GGaussianBlur : public GPipelineStage<cv::GMat>
    {
    public:
        GGaussianBlur(cv::GMat in, GaussianParameters params)
            : GPipelineStage("GaussianBlur", std::move(in))
            , m_params(std::move(params))
        {
            setOutput();
        }

    private:
        virtual void setOutput() override
        {
            m_out = cv::gapi::gaussianBlur(
                m_in,
                cv::Size_<size_t>{ m_params.kernel_size, m_params.kernel_size },
                m_params.sigma
            );
        }

        GaussianParameters m_params = {};
    };

    class GHistEqualize : public GPipelineStage<cv::GMat>
    {
    public:
        GHistEqualize(cv::GMat in)
            : GPipelineStage("HistogramEqualization", std::move(in))
        {
            setOutput();
        }

    protected:
        virtual void setOutput() override
        {
            m_out = cv::gapi::equalizeHist(m_in);
        }
    };

    class GCannyEdges : public GPipelineStage<cv::GMat>
    {
    public:
        GCannyEdges(cv::GMat in, CannyParameters params)
            : GPipelineStage("CannyEdges", std::move(in))
            , m_params(std::move(params))
        {
            setOutput();
        }

    private:
        virtual void setOutput() override
        {
            m_out = cv::gapi::Canny(
                m_in, 
                m_params.low_threshold, 
                m_params.high_threshold
            );
        }

        CannyParameters m_params = {};
    };

    class GContours : public GPipelineStage<cv::GArray<GContour>>
    {
    public:
        GContours(cv::GMat in, ContourParameters params)
            : GPipelineStage("CannyEdges", std::move(in))
            , m_params(std::move(params))
        {
            setOutput();
        }

    private:
        virtual void setOutput() override
        {
            m_out = cv::gapi::findContours(m_in, m_params.retMode, m_params.approxMode);
        }

        ContourParameters m_params = {};
    };

} // namespace mccd

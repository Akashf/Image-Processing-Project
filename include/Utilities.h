#include <memory> 

#include "opencv2/core.hpp"
#include "opencv2/gapi.hpp"


namespace mccd {

    using GContour = cv::GArray<cv::Point>;
    using Contour = std::vector<cv::Point>;
    using Contours = std::vector<Contour>;

    template<typename T>
    using sPtr = std::shared_ptr<T>;

} // namespace mccd
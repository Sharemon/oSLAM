#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace oSLAM
{
    class Frame
    {
    public:
        std::vector<cv::KeyPoint> key_points;
        cv::Mat descriptors;
        std::vector<cv::Point3d> key_points_3d;
        cv::Mat R;
        cv::Mat T;
        cv::Mat rgb;
        cv::Mat depth;
        double time_stamp;
        int idx;
    };
}
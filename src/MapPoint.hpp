#pragma once
#include <opencv2/opencv.hpp>
#include <vector>


namespace oSLAM
{
    class MapPoint
    {
    public:
        cv::KeyPoint key_point;
        cv::Mat descriptor;
        cv::Point3d key_point_3d;
        int match_times;
        int visible_times;
        int frame_cnt;
    };
}
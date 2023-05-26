#pragma once
#include "VisualOdometer.hpp"
#include "MapPoint.hpp"

namespace oSLAM
{
    class MapVisualOdometer : public VisualOdometer
    {
    private:
        cv::Mat R_map, T_map;
        std::vector<MapPoint> map;
        void optimize_map(std::vector<MapPoint>& map);
        void add_to_map(const Frame& frame, std::vector<MapPoint> &map, const std::vector<bool>& mask);
        void pose_estimate_frame_to_map(Frame& cur_frame, std::vector<MapPoint>& map);
        void optimize_pose(std::vector<cv::Point3d> &pts1, const std::vector<cv::Point2d> &pts2, cv::Mat &R, cv::Mat &t);
        bool is_in_frame(const MapPoint &map_point, const Frame &cur_frame);

    public:
        MapVisualOdometer(int max_key_points_num, const cv::Mat& K, double depth_scale, enum VoType type = feature_point);
        ~MapVisualOdometer();
        void add(double timestamp, const cv::Mat &rgb, const cv::Mat &depth);
        void get_all_map_points(std::vector<cv::Point3d>& pts_3d);
        void set_initial_pose(const cv::Mat& R, const cv::Mat& T);
        void get_pose(int frame_idx, cv::Mat &R, cv::Mat &T);
    };
}
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Frame.hpp"

namespace oSLAM
{
 
    class VisualOdometer
    {
    private:
        std::vector<Frame> frames;
        int max_key_points_num;
        double cx, cy, fx, fy;
        double depth_scale;
        std::vector<cv::DMatch> matches;

        void feature_extract(const cv::Mat& rgb, Frame& frame);
        void calc_depth(const cv::Mat& depth, Frame& frame);
        void feature_match(const Frame& ref, const Frame& cur, std::vector<cv::DMatch>& matches);
        void calc_pose_relative(const Frame& ref, Frame& cur, const std::vector<cv::DMatch>& matches);
        void pose_estimation_3d2d(const std::vector<cv::Point3d> &pts1, const std::vector<cv::Point2d> &pts2, cv::Mat &R, cv::Mat &t);
        void pose_estimation_3d3d(const std::vector<cv::Point3d> &pts1, const std::vector<cv::Point3d> &pts2, cv::Mat &R, cv::Mat &t);
    public:
        void add(double timestamp, const cv::Mat &rgb, const cv::Mat& depth);
        void set_pose(int frame_idx, const cv::Mat& R, const cv::Mat& T);
        void get_pose(int frame_idx, cv::Mat& R, cv::Mat& T);
        void get_3d_points(int frame_idx, std::vector<cv::Point3d> &key_points_3d);
        
        VisualOdometer(int max_key_points_num, double cx, double cy, double fx, double fy, double depth_scale);
        ~VisualOdometer();
    };
    
}
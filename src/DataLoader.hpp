#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace oSLAM
{
    class DataLoader
    {
    private:
        std::string data_folder;
        std::string idx_filename;
        std::ifstream idx_filestream;

    public:
        double pop(cv::Mat& rgb, cv::Mat& depth, Eigen::Vector3d& pos, Eigen::Quaterniond& quad);

        DataLoader(std::string data_folder, std::string idx_filename = "gassociate.txt");
        ~DataLoader();
    };   
}
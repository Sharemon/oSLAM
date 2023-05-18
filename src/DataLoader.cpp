#include "DataLoader.hpp"

using namespace std;
using namespace cv;
using namespace oSLAM;


DataLoader::DataLoader(string data_folder, string idx_filename)
{
    DataLoader::data_folder = data_folder + "/";
    DataLoader::idx_filename = data_folder + "/" + idx_filename;

    if (DataLoader::idx_filestream.is_open())
        DataLoader::idx_filestream.close();

    DataLoader::idx_filestream.open(DataLoader::idx_filename, ios::in);
}

DataLoader::~DataLoader()
{
    if (DataLoader::idx_filestream.is_open())
        DataLoader::idx_filestream.close();
}

double DataLoader::pop(Mat& rgb, Mat& depth, Eigen::Vector3d& pos, Eigen::Quaterniond& quad)
{
    if (!DataLoader::idx_filestream.is_open())
        return -1;

    string timestamp_gt, timestamp_rgb, timestamp_depth, filepath_rgb, filepath_depth;
    DataLoader::idx_filestream 
        >> timestamp_gt >> pos(0) >> pos(1) >> pos(2) 
        >> quad.x() >> quad.y() >> quad.z() >> quad.w()
        >> timestamp_rgb >> filepath_rgb >> timestamp_depth >> filepath_depth;

    double timestamp = stod(timestamp_gt);

    rgb = imread(data_folder + filepath_rgb);
    depth = imread(data_folder + filepath_depth, IMREAD_UNCHANGED);

    if (rgb.empty() || depth.empty())
        return -1;

    return timestamp;
}
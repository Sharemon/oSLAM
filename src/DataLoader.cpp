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
    cout << DataLoader::idx_filestream.is_open() <<endl;
}

DataLoader::~DataLoader()
{
    if (DataLoader::idx_filestream.is_open())
        DataLoader::idx_filestream.close();
}

double DataLoader::pop(Mat& rgb, Mat& depth)
{
    if (!DataLoader::idx_filestream.is_open())
        return false;

    string timestamp_rgb, timestamp_depth, filepath_rgb, filepath_depth;
    DataLoader::idx_filestream >> timestamp_rgb >> filepath_rgb >> timestamp_depth >> filepath_depth;

    double timestamp = stod(timestamp_rgb);

    rgb = imread(data_folder + filepath_rgb);
    depth = imread(data_folder + filepath_depth);

    if (rgb.empty() || depth.empty())
        return false;

    return true;
}
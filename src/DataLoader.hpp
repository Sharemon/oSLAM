#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace oSLAM
{
    class DataLoader
    {
    private:
        std::string data_folder;
        std::string idx_filename;
        std::ifstream idx_filestream;

    public:
        double pop(cv::Mat& rgb, cv::Mat& depth);

        DataLoader(std::string data_folder, std::string idx_filename = "associate.txt");
        ~DataLoader();
    };   
}
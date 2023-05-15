#include <vector>
#include <opencv2/opencv.hpp>

namespace oSLAM
{
    class Frame
    {
    public:
        std::vector<cv::KeyPoint> key_points;
        cv::Mat descriptors;
        std::vector<cv::Point2d> key_points_2d;
        std::vector<cv::Point3d> key_points_3d;
        cv::Mat R;
        cv::Mat T;
        double time_stamp;
    };
    

    class Frames
    {
    private:
        std::vector<Frame> frames;
        int max_key_points_num;
    public:
        void add(double timestamp, const cv::Mat &rgb, const cv::Mat& depth);
        void get_pose(int idx, cv::Mat& R, cv::Mat& T);
        
        Frames(int max_key_points_num);
        ~Frames();
    };
    
}
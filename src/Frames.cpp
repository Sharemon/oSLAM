#include "Frames.hpp"

using namespace oSLAM;
using namespace std;
using namespace cv;

Frames::Frames(int max_key_points_num)
{
    Frames::max_key_points_num = max_key_points_num;
}

Frames::~Frames()
{
}

void Frames::add(double timestamp, const Mat& rgb, const Mat& depth)
{
    Frame frame;
    frame.time_stamp = timestamp;

    // 提取rgb图像的orb特征点
    Ptr<ORB> orb_detector = ORB::create();
    orb_detector->detect(rgb, frame.key_points);
    orb_detector->compute(rgb, frame.key_points, frame.descriptors);
    
    //将当前帧加入队列
    Frames::frames.push_back(frame);

    // 如果不是第一帧
    if (Frames::frames.size() > 1)
    {
        // 当前帧与上一帧特征点匹配
        

        // 计算相对位姿关系

    
    }
}

void Frames::get_pose(int idx, Mat& R, Mat& T)
{
    if (Frames::frames.size() <= abs(idx))
    {
        R = Mat();
        T = Mat();
        return;
    }
    else
    {
        ;
    }
}

#include <iostream>
#include <chrono>
#include <unistd.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include "DataLoader.hpp"
#include "VisualOdometer.hpp"
#include "SlamVisualizer.hpp"
#include <pangolin/pangolin.h>
#include <opencv2/core/eigen.hpp>

using namespace oSLAM;
using namespace cv;
using namespace std;

#define GUI_SHOW_RESULT 1

Eigen::Quaterniond R_cvMat_to_eigenQuad(const Mat &R)
{
    Eigen::Matrix3d r;
    cv2eigen(R, r);
    return Eigen::Quaterniond(r);
}

Eigen::Matrix3d R_cvMat_to_eigenMatrix(const Mat &R)
{
    Eigen::Matrix3d r;
    cv2eigen(R, r);
    return r;
}

Eigen::Vector3d T_cvMat_to_eigenVec(const Mat &T)
{
    return Eigen::Vector3d(T.at<double>(0), T.at<double>(1), T.at<double>(2));
}

Mat R_eigenMatrix_to_cvMat(Eigen::Matrix3d r)
{
    Mat R;
    eigen2cv(r, R);
    return R;
}

Mat T_eigenVec_to_cvMat(Eigen::Vector3d t)
{
    Mat T = (Mat_<double>(3, 1) << t(0), t(1), t(2));

    return T;
}

void convert_rgb_depth_to_pointcloud(
    const Mat &rgb, const Mat &depth,
    vector<Eigen::Vector3d>& pts_3d, vector<Eigen::Vector3d>& rgbs,
    const Mat &K, double depth_scale,
    Eigen::Vector3d pos, Eigen::Quaterniond quad)
{
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);

    Eigen::Matrix3d R = quad.toRotationMatrix();

    for (int y = 0; y < rgb.rows; y++)
    {
        for (int x = 0; x < rgb.cols; x++)
        {
            if (depth.at<ushort>(y, x) == 0)
                continue;

            double dis = depth.at<ushort>(y, x) / 5000.0;

            Eigen::Vector3d pt((x - 320.0) / 525.0 * dis, (y - 240.0) / 525.0 * dis, dis);
            Eigen::Vector3d pt_trans = R*pt + pos;
            pts_3d.push_back(pt_trans);
            rgbs.push_back(Eigen::Vector3d(rgb.at<cv::Vec3b>(y,x)[2], rgb.at<cv::Vec3b>(y,x)[1], rgb.at<cv::Vec3b>(y,x)[0]));
        }
    }
}

void convert_3d_keypoints_to_pointcloud(
    const vector<Point3d>& key_pts_3d,
    vector<Eigen::Vector3d>& pts_3d, vector<Eigen::Vector3d>& rgbs,
    Eigen::Vector3d pos, Eigen::Quaterniond quad)
{
    Eigen::Matrix3d R = quad.toRotationMatrix();

    for(auto key_pt_3d : key_pts_3d)
    {
        Eigen::Vector3d pt(key_pt_3d.x, key_pt_3d.y, key_pt_3d.z);
        Eigen::Vector3d pt_trans = R*pt + pos;
        pts_3d.push_back(pt_trans);
        rgbs.push_back(Eigen::Vector3d(255, 255, 255));
    }
}


int main(int argc, char * argv[])
{
    string data_folder = "../data/rgbd_dataset_freiburg1_desk";
    if (argc >= 2)
    {
        data_folder = string(argv[1]);
    }

    DataLoader data_loader(data_folder);
    Mat rgb, depth;
    Eigen::Vector3d gt_pos;
    Eigen::Quaterniond gt_quad;
    double timestamp;
    Mat R, T;
    int cnt = 0;
    double cx = 319.5;
    double cy = 239.5;
    double f = 525;
    double depth_scale = 5000;

    Mat K = (Mat_<double>(3,3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    VisualOdometer vo(500, K, depth_scale, direct);


#if GUI_SHOW_RESULT
    SlamVisualizer visualizer(1504, 960);

    visualizer.initDraw();
    vector<Eigen::Vector3d> traj;
    vector<Eigen::Vector3d> traj_gt;
    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        visualizer.activeAllView();
        visualizer.registerUICallback();

        timestamp = data_loader.pop(rgb, depth, gt_pos, gt_quad);
        if (timestamp < 0)
        {
            break;
        }

        vo.add(timestamp, rgb, depth);
        // 第一帧位姿初值使用gt
        if (cnt == 0)
            vo.set_pose(0, R_eigenMatrix_to_cvMat(gt_quad.toRotationMatrix()), T_eigenVec_to_cvMat(gt_pos));

        if (cnt == 0)
            vo.get_pose(0, R, T);
        else
            vo.get_pose(-1, R, T);

        if (!R.empty())
        {
            cout << "==============" << cnt << "th=============" << endl;
            cout << setprecision(4) << "gt: " << gt_pos.transpose() << ", " << gt_quad.toRotationMatrix().eulerAngles(2,1,0).transpose() << endl;
            cout << setprecision(4) << "vo: " << T_cvMat_to_eigenVec(T).transpose() << ", " << R_cvMat_to_eigenMatrix(R).eulerAngles(2,1,0).transpose() << endl;
        }
        else
        {
            cout << "invalid R,T result" << endl;
        }

        cnt++;

        Eigen::Quaterniond q = R_cvMat_to_eigenQuad(R);
        Eigen::Vector3d t = T_cvMat_to_eigenVec(T);

        traj.push_back(t);
        traj_gt.push_back(gt_pos);

        // 显示数据
        visualizer.displayData(t, q);
        // 绘制轨迹
        visualizer.drawCoordinate();
        visualizer.drawCamWithPose(t, q);
        visualizer.drawTraj(traj);
        visualizer.drawCamWithPose(gt_pos, gt_quad);
        visualizer.drawTrajGt(traj_gt);

        // 画点云
        vector<Point3d> key_pts_3d; 
        vector<Eigen::Vector3d> pts_3d, rgbs;
        //convert_rgb_depth_to_pointcloud(rgb, depth, pts_3d, rgbs, K, depth_scale, gt_pos, gt_quad);
        if (cnt == 0)
            vo.get_3d_points(0, key_pts_3d);
        else
            vo.get_3d_points(-1, key_pts_3d);
        convert_3d_keypoints_to_pointcloud(key_pts_3d, pts_3d, rgbs, gt_pos, gt_quad);
        visualizer.drawPointCloud(pts_3d, rgbs);

        // 画图像
        Mat depth_u8, depth_colormaped;
        depth.convertTo(depth_u8, CV_8UC1, 0.015);
        applyColorMap(depth_u8, depth_colormaped, COLORMAP_JET);
        visualizer.displayImg(rgb, depth_colormaped);

        // 循环
        pangolin::FinishFrame();
    }
#else
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        timestamp = data_loader.pop(rgb, depth, gt_pos, gt_quad);
        if (timestamp < 0)
        {
            break;
        }

        vo.add(timestamp, rgb, depth);
        // 第一帧位姿初值使用gt
        if (cnt == 0)
            vo.set_pose(0, R_eigenMatrix_to_cvMat(gt_quad.toRotationMatrix()), T_eigenVec_to_cvMat(gt_pos));

        if (cnt == 0)
            vo.get_pose(0, R, T);
        else
            vo.get_pose(-1, R, T);

        cnt++;

        if (!R.empty())
        {
            cout << "==============" << cnt << "th=============" << endl;
            cout << setprecision(4) << "gt: " << gt_pos.transpose() << ", " << gt_quad.toRotationMatrix().eulerAngles(2,1,0).transpose() << endl;
            cout << setprecision(4) << "vo: " << T_cvMat_to_eigenVec(T).transpose() << ", " << R_cvMat_to_eigenMatrix(R).eulerAngles(2,1,0).transpose() << endl;
        }
        else
        {
            cout << "invalid R,T result" << endl;
        }

        sleep(1);
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "time elasped " << duration.count() << "s" << endl;
#endif
}

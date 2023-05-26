#include "MapVisualOdometer.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/core/eigen.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
//#include <g2o/types/slam3d/types_slam3d.h>

using namespace oSLAM;
using namespace std;
using namespace cv;

#define KEY_POINT_MAX_NUM (2000)
#define KEY_POINT_MAX_DEPTH (5.0)
#define KEY_POINT_MIN_DEPTH (0.1)

class EdgeProjectXYZ2UVPoseOnly : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectXYZ2UVPoseOnly() {}

    EdgeProjectXYZ2UVPoseOnly(Eigen::Vector3d point, float fx, float fy, float cx, float cy)
        : point_(point), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
    {
    }

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap *v = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d x_local = v->estimate().map(point_);
        double x = x_local[0] * fx_ / x_local[2] + cx_;
        float y = x_local[1] * fy_ / x_local[2] + cy_;
        Eigen::Vector2d preidict_(x, y);

        _error = _measurement - preidict_;
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap *pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(point_);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        double z_2 = z * z;

        _jacobianOplusXi(0, 0) = x * y / z_2 * fx_;
        _jacobianOplusXi(0, 1) = -(1 + (x * x / z_2)) * fx_;
        _jacobianOplusXi(0, 2) = y / z * fx_;
        _jacobianOplusXi(0, 3) = -1. / z * fx_;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = x / z_2 * fx_;

        _jacobianOplusXi(1, 0) = (1 + y * y / z_2) * fy_;
        _jacobianOplusXi(1, 1) = -x * y / z_2 * fy_;
        _jacobianOplusXi(1, 2) = -x / z * fy_;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -1. / z * fy_;
        _jacobianOplusXi(1, 5) = y / z_2 * fy_;
    }

    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &os) const {};

    Eigen::Vector3d point_;
    double cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0; // Camera intrinsics
};

MapVisualOdometer::MapVisualOdometer(int max_key_points_num, const cv::Mat &K, double depth_scale, enum VoType type)
{
    MapVisualOdometer::max_key_points_num = max_key_points_num;
    MapVisualOdometer::cx = K.at<double>(0, 2);
    MapVisualOdometer::cy = K.at<double>(1, 2);
    MapVisualOdometer::fx = K.at<double>(0, 0);
    MapVisualOdometer::fy = K.at<double>(1, 1);
    MapVisualOdometer::depth_scale = depth_scale;
    MapVisualOdometer::type = type; // 在MapVisualOdometer里面，type不起作用

    MapVisualOdometer::R_map = Mat::eye(3, 3, CV_64FC1);
    MapVisualOdometer::T_map = Mat::zeros(3, 1, CV_64FC1);
}

MapVisualOdometer::~MapVisualOdometer()
{
}

bool map_point_compare(MapPoint mp1, MapPoint mp2)
{
    return (mp1.visible_times > mp2.visible_times);
}

void MapVisualOdometer::optimize_map(std::vector<MapPoint> &map)
{
#if 0
    // 当地图点大于1000，则删除被看见的最少的那个点
    sort(map.begin(), map.end(), map_point_compare);

    if (map.size() > KEY_POINT_MAX_NUM)
    {
        map.resize(KEY_POINT_MAX_NUM);
    }
#endif

    for (std::vector<MapPoint>::iterator it = map.begin(); it < map.end();)
    {
        if (it->frame_cnt > 20)
        {
            it = map.erase(it);
        }
        else
        {
            it->frame_cnt++;
            it++;
        }
    }
}

bool MapVisualOdometer::is_in_frame(const MapPoint &map_point, const Frame &cur_frame)
{
    Mat pts_3d = (Mat_<double>(3, 1) << map_point.key_point_3d.x, map_point.key_point_3d.y, map_point.key_point_3d.z);
    Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    Mat pts_3d_pixel = K * (cur_frame.R.inv() * (pts_3d - cur_frame.T));
    double u = pts_3d_pixel.at<double>(0) / pts_3d_pixel.at<double>(2);
    double v = pts_3d_pixel.at<double>(1) / pts_3d_pixel.at<double>(2);

    if (u > 10 && u < cur_frame.rgb.cols - 10 && v > 10 && v < cur_frame.rgb.rows - 10)
    {
        return true;
    }

    return false;
}

void MapVisualOdometer::add_to_map(const Frame &frame, std::vector<MapPoint> &map, const vector<bool> &mask)
{
    for (int i = 0; i < frame.key_points.size(); i++)
    {
        if (mask[i] && frame.key_points_3d[i].z > KEY_POINT_MIN_DEPTH && frame.key_points_3d[i].z < KEY_POINT_MAX_DEPTH)
        {
            MapPoint map_point;
            map_point.visible_times = 0;
            map_point.match_times = 0;
            map_point.frame_cnt = 0;
            map_point.key_point = frame.key_points[i];
            map_point.descriptor = frame.descriptors.row(i).clone();

            Mat pts_3d = (Mat_<double>(3, 1) << frame.key_points_3d[i].x, frame.key_points_3d[i].y, frame.key_points_3d[i].z);
            Mat pts_3d_w = (frame.R * pts_3d + frame.T);

            map_point.key_point_3d = Point3d(pts_3d_w.at<double>(0), pts_3d_w.at<double>(1), pts_3d_w.at<double>(2));

            map.push_back(map_point);
        }
    }
}

void MapVisualOdometer::optimize_pose(std::vector<cv::Point3d> &pts1, const std::vector<cv::Point2d> &pts2, cv::Mat &R, cv::Mat &t)
{
    // 优化位姿和3D点坐标
    // 初始化Tcw
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();

    Eigen::Matrix3d rotation;
    cv2eigen(R, rotation);
    Tcw.rotate(rotation);

    Eigen::Vector3d translation;
    cv2eigen(t, translation);
    Tcw.translate(translation);

#if 0
    // 初始化g2o
    auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto solver_ptr = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr)); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    // 添加边
    g2o::CameraParameters *cam_params = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
    cam_params->setId(0);
    if (!optimizer.addParameter(cam_params))
    {
        assert(false);
    }

    for (int i = 0; i < pts1.size(); i++)
    {
        g2o::VertexPointXYZ *xyz = new g2o::VertexPointXYZ();
        xyz->setId(i+1);
        xyz->setEstimate(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        xyz->setMarginalized(true);
        optimizer.addVertex(xyz);

        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(i);
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(xyz));
        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                              optimizer.vertices().find(0)->second));
        edge->setMeasurement(Eigen::Vector2d(pts2[i].x, pts2[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        edge->setParameterId(0, 0);
        optimizer.addEdge(edge);
    }

    cout << "edges in graph: " << optimizer.edges().size() << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    Tcw = pose->estimate();

    for (int i = 0; i < pts1.size(); i++)
    {
        g2o::VertexPointXYZ *v = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertices().find(i+1)->second);
        Eigen::Vector3d pt = v->estimate();
        pts1[i].x = pt(0);
        pts1[i].y = pt(1);
        pts1[i].z = pt(2);
    }

#else
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> Block; // 求解的向量是6＊1的
    auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
    auto solver_ptr = g2o::make_unique<Block>(std::move(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr)); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    // 添加边
    for (int i = 0; i < pts1.size(); i++)
    {
        EdgeProjectXYZ2UVPoseOnly *edge = new EdgeProjectXYZ2UVPoseOnly(
            Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z),
            fx, fy, cx, cy);
        edge->setVertex(0, pose);
        edge->setMeasurement(Eigen::Vector2d(pts2[i].x, pts2[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setId(i);
        optimizer.addEdge(edge);
    }

    cout << "edges in graph: " << optimizer.edges().size() << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    Tcw = pose->estimate();
#endif    

    // 结果转换
    eigen2cv(Tcw.rotation(), R);
    t = (Mat_<double>(3, 1) << Tcw.translation()(0), Tcw.translation()(1), Tcw.translation()(2));
}

void MapVisualOdometer::pose_estimate_frame_to_map(Frame &cur_frame, std::vector<MapPoint> &map)
{
    // 先从map中选出在视野中的点
    vector<Point3d> pts_3d_in_fov;
    Mat descriptors_in_fov;
    for (auto map_point : map)
    {
        if (is_in_frame(map_point, cur_frame))
        {
            pts_3d_in_fov.push_back(map_point.key_point_3d);
            descriptors_in_fov.push_back(map_point.descriptor);

            map_point.visible_times++;
        }
    }

    cout << "map point num: " << map.size() << endl;
    cout << "point num in fov: " << pts_3d_in_fov.size() << endl;
    cout << "current keypoint num: " << cur_frame.key_points.size() << endl;

    // 地图和当前帧匹配
    vector<DMatch> initial_matches, good_matches;
    BFMatcher matcher(NORM_HAMMING);

    matcher.match(descriptors_in_fov, cur_frame.descriptors, initial_matches);

    if (initial_matches.size() < 20)
        return;

    double min_dis = initial_matches[0].distance;
    for (auto match : initial_matches)
    {
        if (match.distance < min_dis)
            min_dis = match.distance;
    }

    for (auto match : initial_matches)
    {
        if (match.distance <= MAX(min_dis * 2, 30))
            good_matches.push_back(match);
    }

    cout << "good match num: " << good_matches.size() << endl;

    // 根据匹配结果选一下3d地图点和2d当前帧关键点
    vector<bool> kpt_need_to_insert_map(cur_frame.key_points.size(), true);
    vector<Point3d> pts_3d;
    vector<Point2d> pts_2d;
    for (auto match : good_matches)
    {
        pts_3d.push_back(pts_3d_in_fov[match.queryIdx]);
        pts_2d.push_back(cur_frame.key_points[match.trainIdx].pt);

        kpt_need_to_insert_map[match.trainIdx] = false;
    }

    // 其他没有匹配上的点直接加入地图
    add_to_map(cur_frame, map, kpt_need_to_insert_map);

    if (good_matches.size() < 20)
        return;

    // Pnp求个位姿初值
    Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    Mat rvec, tvec;
    solvePnPRansac(pts_3d, pts_2d, K, Mat::zeros(1, 5, CV_64FC1), rvec, tvec);

    Mat R, T;
    Rodrigues(rvec, R);
    T = (Mat_<double>(3, 1) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // 优化位姿和地图点
    optimize_pose(pts_3d, pts_2d, R, T);

    // 相对位姿求绝对位姿
    T = -R.inv() * T;
    R = R.inv();

    cur_frame.R = R.clone();
    cur_frame.T = T.clone();
}

void MapVisualOdometer::add(double timestamp, const cv::Mat &rgb, const cv::Mat &depth)
{
    Frame frame;
    frame.time_stamp = timestamp;
    frame.rgb = rgb.clone();
    frame.depth = depth.clone();
    frame.idx = frames.size();

    // 提取特征点
    feature_extract(rgb, frame);

    // 计算特征点的3d坐标
    calc_depth(depth, frame);

    if (frames.size() == 0)
    {
        frame.R = Mat::eye(3, 3, CV_64FC1);
        frame.T = Mat::zeros(3, 1, CV_64FC1);

        // 第一帧把所有特征点加入map，作为地图点
        vector<bool> mask(frame.key_points.size(), true);
        add_to_map(frame, map, mask);
    }
    else
    {
        // 先用两帧之间的结果作为初值
        // 当前帧与上一帧特征点匹配
        feature_match(
            frames[frames.size() - 1],
            frame,
            matches);

        // 计算相对位姿关系
        calc_pose_from_feature_point(
            frames[frames.size() - 1],
            frame,
            matches, true);

        // 地图和当前帧位姿一起优化
        //pose_estimate_frame_to_map(frame, map);
    }

    // 将当前帧加入队列
    VisualOdometer::frames.push_back(frame);

    // 更新地图
    optimize_map(map);
}

void MapVisualOdometer::get_all_map_points(std::vector<cv::Point3d> &pts_3d)
{
    pts_3d.clear();

    for (auto map_point : map)
    {
        pts_3d.push_back(map_point.key_point_3d);
    }
}

void MapVisualOdometer::set_initial_pose(const cv::Mat &R, const cv::Mat &T)
{
    R_map = R.clone();
    T_map = T.clone();
}

void MapVisualOdometer::get_pose(int frame_idx, cv::Mat &R, cv::Mat &T)
{
    if (frames.size() <= abs(frame_idx))
    {
        R = Mat();
        T = Mat();
        return;
    }
    else
    {
        if (frame_idx >= 0)
        {
            R = R_map * frames[frame_idx].R;
            T = R_map * frames[frame_idx].T + T_map;
        }
        else
        {
            R = R_map * frames[frames.size() + frame_idx].R;
            T = R_map * frames[frames.size() + frame_idx].T + T_map;
        }
    }
}

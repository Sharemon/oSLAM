#include "VisualOdometer.hpp"
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

using namespace oSLAM;
using namespace std;
using namespace cv;
using namespace g2o;

// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
class EdgeSE3ProjectDirect : public BaseUnaryEdge<1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect(Eigen::Vector3d point, float fx, float fy, float cx, float cy, const cv::Mat& image)
        : x_world_(point), fx_(fx), fy_(fy), cx_(cx), cy_(cy), image_(image)
    {
    }

    virtual void computeError()
    {
        const VertexSE3Expmap *v = static_cast<const VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d x_local = v->estimate().map(x_world_);
        float x = x_local[0] * fx_ / x_local[2] + cx_;
        float y = x_local[1] * fy_ / x_local[2] + cy_;
        // check x,y is in the image
        if (x - 4 < 0 || (x + 4) > image_.cols || (y - 4) < 0 || (y + 4) > image_.rows)
        {
            _error(0, 0) = 0.0;
            this->setLevel(1);
        }
        else
        {
            _error(0, 0) = getPixelValue(x, y) - _measurement;
        }
    }

    // plus in manifold
    virtual void linearizeOplus()
    {
        if (level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        VertexSE3Expmap *vtx = static_cast<VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_); // q in book

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        float u = x * fx_ * invz + cx_;
        float v = y * fy_ * invz + cy_;

        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai(0, 0) = -x * y * invz_2 * fx_;
        jacobian_uv_ksai(0, 1) = (1 + (x * x * invz_2)) * fx_;
        jacobian_uv_ksai(0, 2) = -y * invz * fx_;
        jacobian_uv_ksai(0, 3) = invz * fx_;
        jacobian_uv_ksai(0, 4) = 0;
        jacobian_uv_ksai(0, 5) = -x * invz_2 * fx_;

        jacobian_uv_ksai(1, 0) = -(1 + y * y * invz_2) * fy_;
        jacobian_uv_ksai(1, 1) = x * y * invz_2 * fy_;
        jacobian_uv_ksai(1, 2) = x * invz * fy_;
        jacobian_uv_ksai(1, 3) = 0;
        jacobian_uv_ksai(1, 4) = invz * fy_;
        jacobian_uv_ksai(1, 5) = -y * invz_2 * fy_;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv(0, 0) = (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2;
        jacobian_pixel_uv(0, 1) = (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2;

        _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
    }

    // dummy read and write functions because we don't care...
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue(float x, float y)
    {
        uchar *data = &image_.data[int(y) * image_.step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[image_.step] +
            xx * yy * data[image_.step + 1]);
    }

public:
    Eigen::Vector3d x_world_;                 // 3D point in world frame
    double cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0; // Camera intrinsics
    cv::Mat image_;                // reference image
};


class EdgeProjectXYZ2UVPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap >
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



VisualOdometer::VisualOdometer(int max_key_points_num, const cv::Mat &K, double depth_scale, enum VoType type)
{
    VisualOdometer::max_key_points_num = max_key_points_num;
    VisualOdometer::cx = K.at<double>(0, 2);
    VisualOdometer::cy = K.at<double>(1, 2);
    VisualOdometer::fx = K.at<double>(0, 0);
    VisualOdometer::fy = K.at<double>(1, 1);
    VisualOdometer::depth_scale = depth_scale;
    VisualOdometer::type = type;
}


VisualOdometer::VisualOdometer()
{
}

VisualOdometer::~VisualOdometer()
{
}

void VisualOdometer::feature_extract(const cv::Mat &rgb, Frame &frame)
{
    Ptr<ORB> orb_detector = ORB::create(max_key_points_num);
    orb_detector->detect(rgb, frame.key_points);
    orb_detector->compute(rgb, frame.key_points, frame.descriptors);
}

void VisualOdometer::calc_depth(const cv::Mat &depth, Frame &frame)
{
    for (int i = 0; i < frame.key_points.size(); i++)
    {
        double x = frame.key_points[i].pt.x;
        double y = frame.key_points[i].pt.y;

        double dis = depth.at<uint16_t>(int(y), int(x)) / depth_scale;
        frame.key_points_3d.push_back(Point3d((x - cx) / fx * dis, (y - cy) / fy * dis, dis));
    }
}

void VisualOdometer::pose_estimation_3d2d(const std::vector<cv::Point3d> &pts1, const std::vector<cv::Point2d> &pts2, cv::Mat &R, cv::Mat &t, bool use_optimize)
{
    // 利用PnP求解位姿初值
    Mat K = (Mat_<double>(3, 3) << fx, 0, cx,
             0, fy, cy,
             0, 0, 1);

    Mat rvec, tvec;
    solvePnPRansac(pts1, pts2, K, Mat::zeros(1, 5, CV_64FC1), rvec, tvec);

    Rodrigues(rvec, R);
    t = (Mat_<double>(3, 1) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    if (use_optimize)
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
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
        }
        cout << "edges in graph: " << optimizer.edges().size() << endl;
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        Tcw = pose->estimate();

        // 结果转换
        eigen2cv(Tcw.rotation(), R);
        t = (Mat_<double>(3, 1) << Tcw.translation()(0), Tcw.translation()(1), Tcw.translation()(2));
    }
}

void VisualOdometer::pose_estimation_3d3d(const std::vector<cv::Point3d> &pts1, const std::vector<cv::Point3d> &pts2, cv::Mat &R, cv::Mat &t)
{
    Point3d p1(0, 0, 0), p2(0, 0, 0); // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3d(Vec3d(p1) / N);
    p2 = Point3d(Vec3d(p2) / N);
    vector<Point3d> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++)
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0)
    {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2),
         R_(1, 0), R_(1, 1), R_(1, 2),
         R_(2, 0), R_(2, 1), R_(2, 2));
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void VisualOdometer::calc_pose_from_feature_point(const Frame &ref, Frame &cur, const std::vector<cv::DMatch> &matches, bool use_optimize)
{
    vector<Point3d> ref_key_points_3d, cur_key_points_3d;
    vector<Point2d> ref_key_points_2d, cur_key_points_2d;

    // 筛选3D点
    for (auto match : matches)
    {
        Point3d ref_key_point_3d = ref.key_points_3d[match.queryIdx];
        Point3d cur_key_point_3d = cur.key_points_3d[match.trainIdx];

        if (ref_key_point_3d.z == 0 || cur_key_point_3d.z == 0 || ref_key_point_3d.z > 5 || cur_key_point_3d.z > 5)
        {
            continue;
        }

        ref_key_points_3d.push_back(ref_key_point_3d);
        cur_key_points_3d.push_back(cur_key_point_3d);

        ref_key_points_2d.push_back(ref.key_points[match.queryIdx].pt);
        cur_key_points_2d.push_back(cur.key_points[match.trainIdx].pt);
    }

    // 3D点计算位姿
    Mat R, T;
    // pose_estimation_3d3d(cur_key_points_3d, ref_key_points_3d, R, T);
    pose_estimation_3d2d(ref_key_points_3d, cur_key_points_2d, R, T, use_optimize);

    T = -R.inv() * T;
    R = R.inv();

    cur.R = ref.R * R;
    cur.T = ref.R * T + ref.T;
}

void VisualOdometer::pose_estimation_direct(const vector<Measurement> &measurements, const cv::Mat &gray, Eigen::Matrix3d &K, Eigen::Isometry3d &Tcw)
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> DirectBlock; // 求解的向量是6＊1的
    auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<DirectBlock::PoseMatrixType>>();
    auto solver_ptr = g2o::make_unique<DirectBlock>(std::move(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr)); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    // 添加边
    int id = 1;
    for (Measurement m : measurements)
    {
        EdgeSE3ProjectDirect *edge = new EdgeSE3ProjectDirect(
            m.pos,
            K(0, 0), K(1, 1), K(0, 2), K(1, 2), gray);
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);
        optimizer.addEdge(edge);
    }
    cout << "edges in graph: " << optimizer.edges().size() << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    Tcw = pose->estimate();
}

void VisualOdometer::calc_pose_direct(const Frame &ref, Frame &cur)
{
    vector<Measurement> measurements;
    for (int i=0;i<ref.key_points.size();i++)
    {
        Point3d pt_3d = ref.key_points_3d[i];
        float grayscale = ref.key_points[i].response;

        if (pt_3d.z <= 0 || pt_3d.z > 5)
            continue;

        Measurement m(Eigen::Vector3d(pt_3d.x, pt_3d.y, pt_3d.z), grayscale);
        measurements.push_back(m);
    }

    Mat gray;
    cvtColor(cur.rgb, gray, CV_BGR2GRAY);

    Eigen::Matrix3d K_;
    K_ <<   fx, 0, cx, 
            0, fy, cy,
            0,  0,  1;
    
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    pose_estimation_direct(measurements, gray, K_, Tcw);

    Tcw = Tcw.inverse();

    Mat R,T;
    eigen2cv(Tcw.rotation(), R);
    T = (Mat_<double>(3,1) << Tcw.translation()(0),  Tcw.translation()(1),  Tcw.translation()(2));
    
    cur.R = ref.R * R;
    cur.T = ref.R * T + ref.T;
}

void VisualOdometer::feature_match(const Frame &ref, const Frame &cur, std::vector<cv::DMatch> &matches)
{
    vector<DMatch> initial_matches;
    BFMatcher matcher(NORM_HAMMING);

    matcher.match(ref.descriptors, cur.descriptors, initial_matches);

    double min_dis = initial_matches[0].distance;
    for (auto match : initial_matches)
    {
        if (match.distance < min_dis)
            min_dis = match.distance;
    }

    matches.clear();
    for (auto match : initial_matches)
    {
        if (match.distance <= MAX(min_dis * 2, 30))
            matches.push_back(match);
    }
}

void VisualOdometer::add(double timestamp, const Mat &rgb, const Mat &depth)
{
    Frame frame;
    frame.time_stamp = timestamp;
    frame.rgb = rgb.clone();
    frame.depth = depth.clone();

    if (type == feature_point)
    {
        // 提取rgb图像的orb特征点
        VisualOdometer::feature_extract(rgb, frame);

        // 提取关键点的深度信息
        VisualOdometer::calc_depth(depth, frame);

        // 如果不是第一帧
        if (VisualOdometer::frames.size() == 0)
        {
            frame.R = Mat::eye(3, 3, CV_64FC1);
            frame.T = Mat::zeros(3, 1, CV_64FC1);
        }
        else
        {
            // 当前帧与上一帧特征点匹配
            VisualOdometer::feature_match(
                VisualOdometer::frames[VisualOdometer::frames.size() - 1],
                frame,
                VisualOdometer::matches);

            // 计算相对位姿关系
            VisualOdometer::calc_pose_from_feature_point(
                VisualOdometer::frames[VisualOdometer::frames.size() - 1],
                frame,
                VisualOdometer::matches);
        }
    }
    else if (type == direct)
    {
        // 半稠密直接法
        // rgb -> gray
        Mat gray;
        cvtColor(rgb, gray, CV_BGR2GRAY);

        // 选出图像中的关键点（梯度较大的点）
#if 0
        for (int y = 10; y < rgb.rows - 10; y++)
        {
            for (int x = 10; x < rgb.cols - 10; x++)
            {
                double xplus1 = gray.at<uchar>(y, x + 1);
                double xsub1 = gray.at<uchar>(y, x - 1);
                double yplus1 = gray.at<uchar>(y + 1, x);
                double ysub1 = gray.at<uchar>(y - 1, x);
                float grayscale = gray.at<uchar>(y, x);

                if (sqrt(pow(xplus1 - xsub1, 2) + pow(yplus1 - ysub1, 2)) > 150)
                {
                    frame.key_points.push_back(KeyPoint(x, y, 0, 0, grayscale));
                }
            }
        }
#else
        Mat dst, dst_abs;
        cornerHarris(gray, dst, 2, 3, 0.04, BORDER_DEFAULT);
        dst_abs = cv::abs(dst);
        for (int y = 10; y < rgb.rows - 10; y++)
        {
            for (int x = 10; x < rgb.cols - 10; x++)
            {
                if (dst_abs.at<float>(y, x) > 0.001)
                {
                    circle(rgb, Point(x, y), 2, Scalar(0, 255, 0));
                    frame.key_points.push_back(KeyPoint(x, y, 0, 0, gray.at<uchar>(y,x)));
                }
            }
        }
#endif

        // 提取深度信息
        VisualOdometer::calc_depth(depth, frame);

        // 第一帧特殊处理
        if (VisualOdometer::frames.size() == 0)
        {
            frame.R = Mat::eye(3, 3, CV_64FC1);
            frame.T = Mat::zeros(3, 1, CV_64FC1);
        }
        else
        {
            // 直接法计算相机位姿
            VisualOdometer::calc_pose_direct(
                VisualOdometer::frames[VisualOdometer::frames.size() - 1],
                frame);
        }
    }
    else
    {
        ;
    }

    // 将当前帧加入队列
    VisualOdometer::frames.push_back(frame);
}

void VisualOdometer::get_pose(int frame_idx, Mat &R, Mat &T)
{
    if (VisualOdometer::frames.size() <= abs(frame_idx))
    {
        R = Mat();
        T = Mat();
        return;
    }
    else
    {
        if (frame_idx >= 0)
        {
            R = VisualOdometer::frames[frame_idx].R.clone();
            T = VisualOdometer::frames[frame_idx].T.clone();
        }
        else
        {
            R = VisualOdometer::frames[VisualOdometer::frames.size() + frame_idx].R.clone();
            T = VisualOdometer::frames[VisualOdometer::frames.size() + frame_idx].T.clone();
        }
    }
}

void VisualOdometer::set_pose(int frame_idx, const cv::Mat &R, const cv::Mat &T)
{
    if (VisualOdometer::frames.size() <= abs(frame_idx))
    {
        return;
    }
    else
    {
        if (frame_idx >= 0)
        {
            VisualOdometer::frames[frame_idx].R = R.clone();
            VisualOdometer::frames[frame_idx].T = T.clone();
        }
        else
        {
            VisualOdometer::frames[VisualOdometer::frames.size() + frame_idx].R = R.clone();
            VisualOdometer::frames[VisualOdometer::frames.size() + frame_idx].T = T.clone();
        }
    }
}

void VisualOdometer::get_3d_points(int frame_idx, std::vector<cv::Point3d> &key_points_3d)
{
    if (VisualOdometer::frames.size() <= abs(frame_idx))
    {
        key_points_3d.clear();
        return;
    }
    else
    {
        if (frame_idx >= 0)
        {
            key_points_3d = VisualOdometer::frames[frame_idx].key_points_3d;
        }
        else
        {
            key_points_3d = VisualOdometer::frames[VisualOdometer::frames.size() + frame_idx].key_points_3d;
        }
    }
}

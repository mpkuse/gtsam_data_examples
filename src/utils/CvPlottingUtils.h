#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <ostream>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
// using namespace Eigen;
#include <opencv2/core/eigen.hpp>

// using namespace std;

// #include <minitrace.h>
#include <gtest/gtest.h>

class CvPlottingUtils
{
    public: 
    static void say_hi();

    //---------------------------- Conversions ---------------------------------//
    static std::string type2str(int type);
    static std::string cvmat_info( const cv::Mat& mat );
    static std::string eigenmat_info( const Eigen::MatrixXd& mat );
    // static string imgmsg_info(const sensor_msgs::ImageConstPtr &img_msg);
    // static string imgmsg_info(const sensor_msgs::Image& img_msg);
    // static cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

    static std::vector<std::string>
    split( std::string const& original, char separator );


     //---------------------------- Conversions ---------------------------------//
    // convert from opencv format of keypoints to Eigen
    static void keypoint_2_eigen( const std::vector<cv::KeyPoint>& kp, Eigen::MatrixXd& uv, bool make_homogeneous=true );
    static void keypoint_2_eigen( const std::vector<cv::Point2f>& kp, Eigen::MatrixXd& uv, bool make_homogeneous=true );
    static void eigen_2_keypoint( const Eigen::MatrixXd& uv, std::vector<cv::Point2f>& kp );

    // given opencv keypoints and DMatch will produce M1, and M2 the co-ordinates
    static void dmatch_2_eigen( const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                                const std::vector<cv::DMatch> matches,
                                Eigen::MatrixXd& M1, Eigen::MatrixXd& M2,
                                bool make_homogeneous=true
                            );

    static void dmatch_2_cvpoint2f( const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                                const std::vector<cv::DMatch> matches,
                                std::vector<cv::Point2f>& mp1, std::vector<cv::Point2f>& mp2
                            );

    // given a vector of floats converts to Eigen::VectorXd
    static bool vec_of_floats_2_eigen( const std::vector<float>& d_pts, Eigen::VectorXd& d_pts_eigen );
    static bool eigen_2_vec_of_floats( const Eigen::VectorXd& d_pts_eigen, std::vector<float>& d_pts );
    //---------------------------- Conversions ---------------------------------//



    //------------------------------ Annotate an image with points -------------------------//
    
    /// Plots points on an image. 
    /// im: Input image can be eigen gray scale or rgb 
    /// pts_set: A set of points as 3xN or 2xN matrix 
    /// dst: output. Should not be same as im in memory. 
    /// color: the color of the points 
    /// msg: semi-color semarated string, will be printed on the image 
    /// enable_anotations: if this is true, will print numbers along side the marked points.
    /// annotations: if this is empty will print i (the index of the point) next to it 
    ///              if enable_annotations is set. If this is not empty will print annotation(i) 
    ///              as annotation of the i^{th} point.
    static void plot_point_sets( 
    cv::Mat& im, 
    const Eigen::MatrixXd& pts_set, 
    cv::Mat& dst,
    
    const cv::Scalar& color, 
    const std::string& msg = std::string(""), 
    bool enable_anotations = false ,
    const Eigen::VectorXi& annotations = Eigen::VectorXi(),
    const std::vector<bool>& mask = std::vector<bool>()
    );

    //------------------------------- Plot Matchings on image pair -------------------------//

    // Plots [ imA | imaB ] with points correspondences
    // [Input]
    //    imA, imB : Images
    //    ptsA, ptsB : 2xN or 3xN
    //    idxA, idxB : Index of each of the image. This will appear in status part. No other imppact of these.
    //    color_marker : color of the point marker
    //    color_line   : color of the line
    //    annotate_pts : true with putText for each point. False will not putText.
    // [Output]
    //    outImg : Output image
    static void plot_point_pair( const cv::Mat& imA, const Eigen::MatrixXd& ptsA, int idxA,
                          const cv::Mat& imB, const Eigen::MatrixXd& ptsB, int idxB,
                          cv::Mat& dst,
                          const cv::Scalar& color_marker,
                          const cv::Scalar& color_line=cv::Scalar(0,255,0),
                          bool annotate_pts=false,
                          const std::string& msg=std::string("N.A")
                         );


};
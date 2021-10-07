/* Find the SE(2) rotation between two images.

    Load an image. Inplane rotate it with opencv function.
    Detect point features on the two images and setup the
    optimization problem. Recover the rotation.
*/

#include <iostream>

// OpenCV
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

// GTSam
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "../custom_factors/ImageRotationFactor.h"

#include "utils/CvPlottingUtils.h"
#include "utils/TermColor.h"

class PrepareData {
public:
  PrepareData() {
    std::cout << "---------------------------PrepareData-----------------------"
                 "-----\n";
    //-- Load Image
    std::cout << "-- Load Test Image\n";
    test_im = cv::imread("../data/rgb-d/rgb/1.png");
    EXPECT_FALSE(test_im.empty());
    std::cout << "test_im: " << CvPlottingUtils::cvmat_info(test_im)
              << std::endl;

    //-- Prepare rotation matrix
    std::cout << "-- Prepare rotation matrix\n";
    // cv::Mat matRotation = cv::getRotationMatrix2D(cv::Point2f(0, 0),
    // 30, 1.0);
    this->matRotation  = cv::getRotationMatrix2D(
        cv::Point2f(test_im.cols / 2, test_im.rows / 2), 20, 1.0);
    
    std::cout << "matRotation: " << matRotation << std::endl;

    //-- Rotate Image with opencv
    cv::warpAffine(test_im, test_im_rotated, matRotation, test_im.size());

    cv::cvtColor(test_im, test_im_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(test_im_rotated, test_im_rotated_gray, cv::COLOR_BGR2GRAY);

    //-- Extract Features
    std::cout << "-- Extract Features\n";
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(test_im_gray, cv::Mat(), keypoints1,
                               descriptors1);
    detector->detectAndCompute(test_im_rotated_gray, cv::Mat(), keypoints2,
                               descriptors2);
    std::cout << "keypoints1: " << keypoints1.size() << "\t";
    std::cout << "keypoints2: " << keypoints2.size() << "\t" << std::endl;
    std::cout << "descriptors1: " << CvPlottingUtils::cvmat_info(descriptors1)
              << "\t";
    std::cout << "descriptors2: " << CvPlottingUtils::cvmat_info(descriptors2)
              << "\n";

    //-- Match Features
    std::cout << "-- Match Features\n";
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch>
        good_matches; // result of nearest neighbour + Lowe's ratio test
    good_matches.clear();
    for (size_t i = 0; i < knn_matches.size(); i++) {
      if (knn_matches[i][0].distance <
          ratio_thresh * knn_matches[i][1].distance) {
        good_matches.push_back(knn_matches[i][0]);
      }
    }
    std::cout << "#good_matches (after lowe ratio test): "
              << good_matches.size() << std::endl;

    //-- Fundamental matrix test
    //---- FindFundamentaMatrix Ransac
    //---- get rid of correspondences which dont satisfy fundamental matrix

    std::vector<cv::Point2f> matched_1, matched_2;
    std::vector<uchar> status;
    status.clear();
    if (good_matches.size() > 20) {
      CvPlottingUtils::dmatch_2_cvpoint2f(keypoints1, keypoints2, good_matches,
                                          matched_1, matched_2);
      cv::findFundamentalMat(matched_1, matched_2, cv::FM_RANSAC, 3.0, 0.99,
                             status);
    }

    std::vector<cv::Point2f> matched_1_filtered, matched_2_filtered;
    matched_1_filtered.clear();
    matched_2_filtered.clear();
    for (auto y = 0; y < status.size(); y++) {
      if (status.at(y) > 0u) {
        matched_1_filtered.push_back(matched_1.at(y));
        matched_2_filtered.push_back(matched_2.at(y));
      }
    }
    std::cout << "#matches (after fundamental matrix test): "
              << matched_1_filtered.size() << std::endl;
    this->test_im_matched_feat = matched_1_filtered;
    this->test_im_rotated_matched_feat = matched_2_filtered;

    //-- Plot matches
    Eigen::MatrixXd eigen_matched_1_filtered, eigen_matched_2_filtered;
    CvPlottingUtils::keypoint_2_eigen(matched_1_filtered,
                                      eigen_matched_1_filtered);
    CvPlottingUtils::keypoint_2_eigen(matched_2_filtered,
                                      eigen_matched_2_filtered);
    cv::Mat dst;
    CvPlottingUtils::plot_point_pair(
        test_im_gray, eigen_matched_1_filtered, 0, test_im_rotated_gray,
        eigen_matched_2_filtered, 0, dst, cv::Scalar(0, 0, 255));
    cv::imshow("matched features", dst);

    std::cout << "---------------------------PrepareData-----------------------"
                 "-----\n";
  }

public:
  cv::Mat test_im, test_im_rotated;
  cv::Mat test_im_gray, test_im_rotated_gray;
  cv::Mat matRotation ;

  std::vector<cv::Point2f> test_im_matched_feat, test_im_rotated_matched_feat;
};




int main() {
  std::cout << "Fit Image Rotation\n";

  //-- Data
  PrepareData d;

  {
    Eigen::MatrixXd matRotation_eigen;
    cv::cv2eigen( d.matRotation, matRotation_eigen );
    Eigen::MatrixXd matRotation_eigen2 = Eigen::MatrixXd::Identity(3,3);
    matRotation_eigen2 << matRotation_eigen, 0, 0, 1;
    auto matRotation_gtsam = gtsam::Pose2( matRotation_eigen2 );
    matRotation_gtsam.print( "matRotation_gtsam (GROUND TRUTH): " );
  }

  //-- Create an empty nonlinear factor graph
  gtsam::NonlinearFactorGraph graph;

  
  //-- Add a factor for each measurement
  auto measurement_noise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(2.0, 2.0));
  std::cout << "n_ ImageRotationFactor added: " << d.test_im_matched_feat.size() << std::endl;
  for (int i = 0; i < /*20*/ d.test_im_matched_feat.size(); i++) {
    gtsam::Point2 X(d.test_im_matched_feat.at(i).x,
                    d.test_im_matched_feat.at(i).y);
    gtsam::Point2 Xd(d.test_im_rotated_matched_feat.at(i).x,
                     d.test_im_rotated_matched_feat.at(i).y);

    // std::cout << X << "<-->" << Xd << std::endl;
    auto factor =
        ImageRotationFactor(measurement_noise, gtsam::Symbol('T'), X, Xd);
    graph.push_back(factor);
  }

  //-- Initial Guess
  gtsam::Values initial;
  initial.print("Initial Guess:\n");
  initial.insert(gtsam::Symbol('T'), gtsam::Pose2(-60, 200.0, 0.2));
  

  //-- Solve and Retrive Result
  gtsam::Values result =
      gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();
  result.print("Final Result:\n");
  gtsam::Pose2 result_pose2 = result.at<gtsam::Pose2>(gtsam::Symbol('T'));
  //   std::cout << "result_pose2: " << result_pose2 << std::endl;
  result_pose2.print("result_pose2");


  cv::imshow("test_im", d.test_im);
  cv::imshow("test_im_rotated", d.test_im_rotated);
  cv::waitKey(0);
}
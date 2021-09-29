/**
 * A Super basic GTSAM example
 * */

#include <iostream>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>



int main() {
  std::cout << "Hello GTSAM\n";

  // Create an empty nonlinear factor graph
  gtsam::NonlinearFactorGraph graph;

  // Add a prior on the first pose, setting it to the origin
  // A prior factor consists of a mean and a noise model (covariance matrix)
  gtsam::Pose2 priorMean(5.0, 0.0, 0.0); // prior at origin
  auto priorNoise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose2>>( gtsam::Symbol('x', 1), priorMean,
                                                         priorNoise);

  // Add odometry factors
  gtsam::Pose2 odometry(2.0, 0.0, 0.0);
  // For simplicity, we will use the same noise model for each odometry factor
  //   gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
  auto odometryNoise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.2, 0.2, 0.1));
  // Create odometry (Between) factors between consecutive poses
  graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2>>(gtsam::Symbol('x', 1), gtsam::Symbol('x', 2), odometry,
                                                           odometryNoise);
  graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2>>(gtsam::Symbol('x', 2), gtsam::Symbol('x', 3), odometry,
                                                           odometryNoise);
  graph.print("\nFactor Graph:\n"); // print

  
// Create the data structure to hold the initialEstimate estimate to the
// solution For illustrative purposes, these have been deliberately set to
// incorrect values
gtsam::Values initial;
initial.insert(gtsam::Symbol('x',1), gtsam::Pose2(5.5, 0.0, 0.2));
initial.insert(gtsam::Symbol('x',2), gtsam::Pose2(7.3, 0.1, -0.2));
initial.insert(gtsam::Symbol('x',3), gtsam::Pose2(9.1, 0.1, 0.1));
initial.print("\nInitial Estimate:\n"); // print

// optimize using Levenberg-Marquardt optimization
gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();
result.print("Final Result:\n");

}
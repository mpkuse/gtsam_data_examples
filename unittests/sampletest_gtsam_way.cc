#include <iostream>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <CppUnitLite/TestHarness.h>

TEST(SampleTests, Unit3) {
  int x = 3;
  EXPECT(true);
}

TEST(SampleTests, PriorFactor) {
  gtsam::Pose2 priorMean(0.0, 0.0, 0.0); // prior at origin
  gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));

  auto factor = gtsam::PriorFactor<gtsam::Pose2>(gtsam::Symbol('x'), priorMean,
                                                 priorNoise);

  gtsam::Values values;
  values.insert(gtsam::Symbol('x'), gtsam::Pose2(0.000, 0.00, 0.057));

  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-7, 1e-5);
}
#if 1
TEST(SampleTests, BetweenFactor) {
  gtsam::Rot3 R1 = gtsam::Rot3::Rodrigues(0.1, 0.2, 0.3);
  gtsam::Rot3 R2 = gtsam::Rot3::Rodrigues(0.4, 0.5, 0.6);
  gtsam::Rot3 noise = gtsam::Rot3(); // Rot3::Rodrigues(0.01, 0.01, 0.01); //
                                     // Uncomment to make unit test fail
  gtsam::Rot3 measured = R1.between(R2) * noise;

  auto priorModel = gtsam::noiseModel::Diagonal::Variances(
      (gtsam::Vector(3) << 1e-4, 1e-4, 1e-4).finished());

  //   gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
  //   gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));

  auto factor = gtsam::BetweenFactor<gtsam::Rot3>(
      gtsam::Symbol('x', 1), gtsam::Symbol('x', 2), measured, priorModel);

  gtsam::Values values;
  values.insert(gtsam::Symbol('x', 1 ), R1 );
  values.insert(gtsam::Symbol('x', 2 ), R2 );

  // this will make the test fail, gtsam only passes the test at the exact same point
//   values.insert(gtsam::Symbol('x', 1 ), gtsam::Rot3::Rodrigues(0.14, 0.22, 0.31) );
//   values.insert(gtsam::Symbol('x', 2 ), gtsam::Rot3::Rodrigues(0.45, 0.55, 0.63) );

  EXPECT_CORRECT_FACTOR_JACOBIANS( factor, values, 1e-7, 1e-5 );
}
#endif 

int main(int argc, char **argv) {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}

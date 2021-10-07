#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>

/// ImageRotationFactor
/// Measurements: Point2 X, Point2 Xd
/// Optimization variable: T. The transform that is to be applied on Xd to bring it to X.
/// Cost function: \sum_i || T * Xd_i - X_i ||
class ImageRotationFactor : public gtsam::NoiseModelFactor1<gtsam::Pose2> {
  typedef NoiseModelFactor1<gtsam::Pose2> Base;

  // measurements
  gtsam::Point2 X_i, Xd_i;

public:
  ImageRotationFactor(const gtsam::SharedNoiseModel &model,
                      const gtsam::Key &key, const gtsam::Point2 _X_i,
                      const gtsam::Point2 _Xd_i)
      : Base(model, key), X_i(_X_i), Xd_i(_Xd_i) {}

  // evaluate error
  gtsam::Vector
  evaluateError(const gtsam::Pose2 &Tr,
                boost::optional<gtsam::Matrix &> H = boost::none) const {
    return Tr.transform_to(Xd_i, H) - X_i;
  }
};
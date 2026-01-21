// Simple ROI projector library.
// Load calibration from JSON, project ROI using depth.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace roi_projector {

struct Point2D {
  double u = 0.0;
  double v = 0.0;
};

struct Point3D {
  double u = 0.0;
  double v = 0.0;
  double z = 0.0;
};

struct Rect {
  double x = 0.0;
  double y = 0.0;
  double w = 0.0;
  double h = 0.0;
};

struct CornersResult {
  bool ok = false;
  std::array<Point2D, 4> points{};
  std::string message;
};

bool IsRoiInsideQuad(const std::array<Point2D, 4>& quad, const std::array<Point2D, 4>& barcode);

class Projector {
 public:
  bool LoadCalibration(const std::string& file_path);
  CornersResult ProjectCorners(const std::array<Point3D, 4>& corners) const;

 private:
  bool has_calibration_ = false;
  std::array<std::array<double, 4>, 4> extrinsic_{};   // 4x4
  std::array<std::array<double, 3>, 3> camera1_{};     // 3x3
  std::array<std::array<double, 3>, 3> camera2_{};     // 3x3
  std::array<double, 5> dist1_{};                      // k1,k2,p1,p2,k3
  std::array<double, 5> dist2_{};                      // k1,k2,p1,p2,k3

  bool ParseMatrix4x4(const std::string& json, const std::string& key,
                      std::array<std::array<double, 4>, 4>& out) const;
  bool ParseMatrix3x3(const std::string& json, const std::string& key,
                      std::array<std::array<double, 3>, 3>& out) const;
  bool ParseDistortion5(const std::string& json, const std::string& key,
                        std::array<double, 5>& out) const;
  bool ParseNumberArray(const std::string& json, size_t start_pos,
                        size_t expected_count, std::vector<double>& out) const;
  bool FindKeyArrayStart(const std::string& json, const std::string& key,
                         size_t& start_pos) const;

  bool TransformPoint(double u, double v, double depth,
                      double& out_u, double& out_v) const;
  bool HasDistortion(const std::array<double, 5>& dist) const;
  void UndistortNormalized(double xd, double yd,
                           const std::array<double, 5>& dist,
                           double& xu, double& yu) const;
  void DistortNormalized(double x, double y,
                         const std::array<double, 5>& dist,
                         double& xd, double& yd) const;
};

}  // namespace roi_projector

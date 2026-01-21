#include <iostream>

#include "roi_projector.h"

int main(int argc, char** argv) {
  const std::string calib_path = (argc > 1) ? argv[1] : "test/calib_out.json";

  roi_projector::Projector projector;
  if (!projector.LoadCalibration(calib_path)) {
    std::cerr << "Failed to load calibration: " << calib_path << "\n";
    return 1;
  }

  std::array<roi_projector::Point3D, 4> corners{};
  if (argc >= 14) {
    int arg = 2;
    for (size_t i = 0; i < corners.size(); ++i) {
      corners[i].u = std::stod(argv[arg++]);
      corners[i].v = std::stod(argv[arg++]);
      corners[i].z = std::stod(argv[arg++]);
    }
  } else {
    std::cout << "Usage: roi_projector_test.exe <calib.json> "
                 "<u1> <v1> <z1> <u2> <v2> <z2> <u3> <v3> <z3> <u4> <v4> <z4>\n";
    std::cout << "Corners not provided, using default corners.\n";
    corners[0] = {100.0, 200.0, 1000.0};
    corners[1] = {400.0, 200.0, 1000.0};
    corners[2] = {400.0, 350.0, 1000.0};
    corners[3] = {100.0, 350.0, 1000.0};
  }

  const auto result = projector.ProjectCorners(corners);
  if (!result.ok) {
    std::cerr << "Project corners failed: " << result.message << "\n";
    return 1;
  }

  std::cout << "Projected corners:\n";
  for (size_t i = 0; i < result.points.size(); ++i) {
    std::cout << "  [" << i << "] u=" << result.points[i].u
              << " v=" << result.points[i].v << "\n";
  }
  return 0;
}

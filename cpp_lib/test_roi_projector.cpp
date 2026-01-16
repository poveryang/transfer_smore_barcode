#include <iostream>

#include "roi_projector.h"

int main(int argc, char** argv) {
  const std::string calib_path = (argc > 1) ? argv[1] : "test/calib_out.json";
  const double depth = (argc > 2) ? std::stod(argv[2]) : 1000.0;

  roi_projector::Projector projector;
  if (!projector.LoadCalibration(calib_path)) {
    std::cerr << "Failed to load calibration: " << calib_path << "\n";
    return 1;
  }

  roi_projector::Roi roi;
  if (argc >= 7) {
    roi.x = std::stod(argv[3]);
    roi.y = std::stod(argv[4]);
    roi.w = std::stod(argv[5]);
    roi.h = std::stod(argv[6]);
  } else {
    std::cout << "Usage: roi_projector_test.exe <calib.json> <depth> <x> <y> <w> <h>\n";
    std::cout << "ROI not provided, using default ROI: (100, 200, 300, 150)\n";
    roi.x = 100;
    roi.y = 200;
    roi.w = 300;
    roi.h = 150;
  }

  const auto result = projector.ProjectRoi(roi, depth);
  if (!result.ok) {
    std::cerr << "Project ROI failed: " << result.message << "\n";
    return 1;
  }

  std::cout << "Projected ROI: x=" << result.roi.x
            << " y=" << result.roi.y
            << " w=" << result.roi.w
            << " h=" << result.roi.h << "\n";
  return 0;
}

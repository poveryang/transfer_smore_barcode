// Simple ROI projector library.
#include "roi_projector.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace roi_projector {

namespace {

std::string ReadAllText(const std::string& path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return std::string();
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

bool Projector::LoadCalibration(const std::string& file_path) {
  const std::string json = ReadAllText(file_path);
  if (json.empty()) {
    return false;
  }

  if (!ParseMatrix4x4(json, "extrinsic_matrix", extrinsic_)) {
    return false;
  }
  if (!ParseMatrix3x3(json, "camera1_matrix", camera1_)) {
    return false;
  }
  if (!ParseMatrix3x3(json, "camera2_matrix", camera2_)) {
    return false;
  }
  ParseDistortion5(json, "camera1_distortion", dist1_);
  ParseDistortion5(json, "camera2_distortion", dist2_);

  has_calibration_ = true;
  return true;
}

RoiResult Projector::ProjectRoi(const Roi& roi, double depth) const {
  RoiResult result;
  if (!has_calibration_) {
    result.message = "calibration not loaded";
    return result;
  }
  if (depth <= 0.0 || !std::isfinite(depth)) {
    result.message = "invalid depth";
    return result;
  }

  const double x = roi.x;
  const double y = roi.y;
  const double w = roi.w;
  const double h = roi.h;

  const std::array<std::pair<double, double>, 4> corners = {
      std::make_pair(x, y),
      std::make_pair(x + w, y),
      std::make_pair(x + w, y + h),
      std::make_pair(x, y + h),
  };

  std::vector<std::pair<double, double>> projected;
  projected.reserve(4);
  for (const auto& pt : corners) {
    double out_u = 0.0;
    double out_v = 0.0;
    if (TransformPoint(pt.first, pt.second, depth, out_u, out_v)) {
      projected.emplace_back(out_u, out_v);
    }
  }

  if (projected.size() < 2) {
    result.message = "not enough valid projected points";
    return result;
  }

  double min_x = projected[0].first;
  double max_x = projected[0].first;
  double min_y = projected[0].second;
  double max_y = projected[0].second;
  for (const auto& pt : projected) {
    min_x = std::min(min_x, pt.first);
    max_x = std::max(max_x, pt.first);
    min_y = std::min(min_y, pt.second);
    max_y = std::max(max_y, pt.second);
  }

  result.ok = true;
  result.roi.x = min_x;
  result.roi.y = min_y;
  result.roi.w = max_x - min_x;
  result.roi.h = max_y - min_y;
  result.message = "ok";
  return result;
}

bool Projector::TransformPoint(double u, double v, double depth,
                               double& out_u, double& out_v) const {
  const double fx1 = camera1_[0][0];
  const double fy1 = camera1_[1][1];
  const double cx1 = camera1_[0][2];
  const double cy1 = camera1_[1][2];

  double x_norm = (u - cx1) / fx1;
  double y_norm = (v - cy1) / fy1;
  if (HasDistortion(dist1_)) {
    double xu = 0.0;
    double yu = 0.0;
    UndistortNormalized(x_norm, y_norm, dist1_, xu, yu);
    x_norm = xu;
    y_norm = yu;
  }

  const double x = x_norm * depth;
  const double y = y_norm * depth;
  const double z = depth;

  const double x2 = extrinsic_[0][0] * x + extrinsic_[0][1] * y +
                    extrinsic_[0][2] * z + extrinsic_[0][3];
  const double y2 = extrinsic_[1][0] * x + extrinsic_[1][1] * y +
                    extrinsic_[1][2] * z + extrinsic_[1][3];
  const double z2 = extrinsic_[2][0] * x + extrinsic_[2][1] * y +
                    extrinsic_[2][2] * z + extrinsic_[2][3];

  if (z2 <= 0.0 || !std::isfinite(z2)) {
    return false;
  }

  const double fx2 = camera2_[0][0];
  const double fy2 = camera2_[1][1];
  const double cx2 = camera2_[0][2];
  const double cy2 = camera2_[1][2];

  double x2_norm = x2 / z2;
  double y2_norm = y2 / z2;
  if (HasDistortion(dist2_)) {
    double xd = 0.0;
    double yd = 0.0;
    DistortNormalized(x2_norm, y2_norm, dist2_, xd, yd);
    x2_norm = xd;
    y2_norm = yd;
  }

  out_u = fx2 * x2_norm + cx2;
  out_v = fy2 * y2_norm + cy2;
  return std::isfinite(out_u) && std::isfinite(out_v);
}

bool Projector::FindKeyArrayStart(const std::string& json,
                                  const std::string& key,
                                  size_t& start_pos) const {
  const std::string quoted = "\"" + key + "\"";
  const size_t key_pos = json.find(quoted);
  if (key_pos == std::string::npos) {
    return false;
  }
  const size_t array_pos = json.find('[', key_pos);
  if (array_pos == std::string::npos) {
    return false;
  }
  start_pos = array_pos;
  return true;
}

bool Projector::ParseNumberArray(const std::string& json, size_t start_pos,
                                 size_t expected_count,
                                 std::vector<double>& out) const {
  out.clear();
  if (start_pos >= json.size() || json[start_pos] != '[') {
    return false;
  }

  size_t i = start_pos;
  int depth = 0;
  while (i < json.size()) {
    const char c = json[i];
    if (c == '[') {
      depth++;
      i++;
      continue;
    }
    if (c == ']') {
      depth--;
      i++;
      if (depth == 0) {
        break;
      }
      continue;
    }
    if (c == '-' || c == '+' || std::isdigit(static_cast<unsigned char>(c)) ||
        c == '.') {
      char* end_ptr = nullptr;
      const double value = std::strtod(json.c_str() + i, &end_ptr);
      if (end_ptr == json.c_str() + i) {
        return false;
      }
      out.push_back(value);
      i = static_cast<size_t>(end_ptr - json.c_str());
      continue;
    }
    i++;
  }

  if (out.size() < expected_count) {
    return false;
  }
  if (out.size() > expected_count) {
    out.resize(expected_count);
  }
  return true;
}

bool Projector::ParseMatrix4x4(
    const std::string& json, const std::string& key,
    std::array<std::array<double, 4>, 4>& out) const {
  size_t start_pos = 0;
  if (!FindKeyArrayStart(json, key, start_pos)) {
    return false;
  }
  std::vector<double> values;
  if (!ParseNumberArray(json, start_pos, 16, values)) {
    return false;
  }
  for (size_t r = 0; r < 4; ++r) {
    for (size_t c = 0; c < 4; ++c) {
      out[r][c] = values[r * 4 + c];
    }
  }
  return true;
}

bool Projector::ParseMatrix3x3(
    const std::string& json, const std::string& key,
    std::array<std::array<double, 3>, 3>& out) const {
  size_t start_pos = 0;
  if (!FindKeyArrayStart(json, key, start_pos)) {
    return false;
  }
  std::vector<double> values;
  if (!ParseNumberArray(json, start_pos, 9, values)) {
    return false;
  }
  for (size_t r = 0; r < 3; ++r) {
    for (size_t c = 0; c < 3; ++c) {
      out[r][c] = values[r * 3 + c];
    }
  }
  return true;
}

bool Projector::ParseDistortion5(
    const std::string& json, const std::string& key,
    std::array<double, 5>& out) const {
  out.fill(0.0);
  size_t start_pos = 0;
  if (!FindKeyArrayStart(json, key, start_pos)) {
    return false;
  }
  std::vector<double> values;
  if (!ParseNumberArray(json, start_pos, 5, values)) {
    return false;
  }
  for (size_t i = 0; i < 5; ++i) {
    out[i] = values[i];
  }
  return true;
}

bool Projector::HasDistortion(const std::array<double, 5>& dist) const {
  for (double v : dist) {
    if (v != 0.0) {
      return true;
    }
  }
  return false;
}

void Projector::UndistortNormalized(double xd, double yd,
                                    const std::array<double, 5>& dist,
                                    double& xu, double& yu) const {
  const double k1 = dist[0];
  const double k2 = dist[1];
  const double p1 = dist[2];
  const double p2 = dist[3];
  const double k3 = dist[4];

  double x = xd;
  double y = yd;
  for (int i = 0; i < 5; ++i) {
    const double r2 = x * x + y * y;
    const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    const double x_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double y_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    x = (xd - x_t) / radial;
    y = (yd - y_t) / radial;
  }
  xu = x;
  yu = y;
}

void Projector::DistortNormalized(double x, double y,
                                  const std::array<double, 5>& dist,
                                  double& xd, double& yd) const {
  const double k1 = dist[0];
  const double k2 = dist[1];
  const double p1 = dist[2];
  const double p2 = dist[3];
  const double k3 = dist[4];

  const double r2 = x * x + y * y;
  const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const double x_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const double y_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

  xd = x * radial + x_t;
  yd = y * radial + y_t;
}

}  // namespace roi_projector

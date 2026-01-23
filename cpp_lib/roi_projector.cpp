// Simple ROI projector library.
#include "roi_projector.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

bool IsPointInConvexQuad(const std::array<roi_projector::Point2D, 4>& quad,
                         const roi_projector::Point2D& p) {
  constexpr double kEps = 1e-9;
  int sign = 0;
  for (size_t i = 0; i < quad.size(); ++i) {
    const auto& a = quad[i];
    const auto& b = quad[(i + 1) % quad.size()];
    const double cross = (b.u - a.u) * (p.v - a.v) - (b.v - a.v) * (p.u - a.u);
    if (std::fabs(cross) <= kEps) {
      continue;
    }
    const int current_sign = (cross > 0.0) ? 1 : -1;
    if (sign == 0) {
      sign = current_sign;
    } else if (sign != current_sign) {
      return false;
    }
  }
  return sign != 0;
}

// 计算多边形面积（使用鞋带公式）
double ComputePolygonArea(const std::vector<roi_projector::Point2D>& polygon) {
  if (polygon.size() < 3) {
    return 0.0;
  }
  double area = 0.0;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const size_t j = (i + 1) % polygon.size();
    area += polygon[i].u * polygon[j].v;
    area -= polygon[j].u * polygon[i].v;
  }
  return std::fabs(area) / 2.0;
}

// 计算线段与半平面的交点
roi_projector::Point2D ComputeLineIntersection(
    const roi_projector::Point2D& p1, const roi_projector::Point2D& p2,
    const roi_projector::Point2D& clip_p1, const roi_projector::Point2D& clip_p2) {
  const double dx1 = p2.u - p1.u;
  const double dy1 = p2.v - p1.v;
  const double dx2 = clip_p2.u - clip_p1.u;
  const double dy2 = clip_p2.v - clip_p1.v;
  
  const double denom = dx1 * dy2 - dy1 * dx2;
  if (std::fabs(denom) < 1e-9) {
    // 平行线，返回中点
    return {(p1.u + p2.u) / 2.0, (p1.v + p2.v) / 2.0};
  }
  
  const double t = ((p1.u - clip_p1.u) * dy2 - (p1.v - clip_p1.v) * dx2) / denom;
  return {p1.u + t * dx1, p1.v + t * dy1};
}

// 判断点是否在半平面内（相对于裁剪边）
bool IsPointInsideHalfPlane(const roi_projector::Point2D& p,
                            const roi_projector::Point2D& clip_p1,
                            const roi_projector::Point2D& clip_p2) {
  const double cross = (clip_p2.u - clip_p1.u) * (p.v - clip_p1.v) -
                       (clip_p2.v - clip_p1.v) * (p.u - clip_p1.u);
  return cross >= 0.0;  // 顺时针方向，点在左侧或线上
}

// 使用 Sutherland-Hodgman 算法计算两个凸多边形的交集
// 返回 poly1 在 poly2 内部的部分（即 poly1 ∩ poly2）
std::vector<roi_projector::Point2D> ComputeConvexPolygonIntersection(
    const std::array<roi_projector::Point2D, 4>& poly1,
    const std::array<roi_projector::Point2D, 4>& poly2) {
  std::vector<roi_projector::Point2D> result;
  // 将 poly1 转换为 vector
  for (const auto& pt : poly1) {
    result.push_back(pt);
  }
  
  // 使用 poly2 的每条边作为裁剪边来裁剪 poly1
  for (size_t i = 0; i < poly2.size(); ++i) {
    const auto& clip_p1 = poly2[i];
    const auto& clip_p2 = poly2[(i + 1) % poly2.size()];
    
    std::vector<roi_projector::Point2D> new_result;
    if (result.empty()) {
      break;
    }
    
    // 处理闭合循环：从最后一个点开始，遍历到第一个点
    const roi_projector::Point2D* prev = &result.back();
    bool prev_inside = IsPointInsideHalfPlane(*prev, clip_p1, clip_p2);
    
    for (size_t j = 0; j < result.size(); ++j) {
      const auto& curr = result[j];
      bool curr_inside = IsPointInsideHalfPlane(curr, clip_p1, clip_p2);
      
      if (curr_inside) {
        if (!prev_inside) {
          // 从外部进入，添加交点
          roi_projector::Point2D intersection = ComputeLineIntersection(*prev, curr, clip_p1, clip_p2);
          new_result.push_back(intersection);
        }
        new_result.push_back(curr);
      } else if (prev_inside) {
        // 从内部出去，添加交点
        roi_projector::Point2D intersection = ComputeLineIntersection(*prev, curr, clip_p1, clip_p2);
        new_result.push_back(intersection);
      }
      
      prev = &curr;
      prev_inside = curr_inside;
    }
    
    // 如果结果为空，说明没有交集
    if (new_result.empty()) {
      std::cout << "[IOU Debug] Clip edge " << i << " resulted in empty intersection" << std::endl << std::flush;
      return std::vector<roi_projector::Point2D>();
    }
    
    result = std::move(new_result);
  }
  
  return result;
}

// 计算两个四边形的 IOU（内部实现）
double ComputeIOUImpl(const std::array<roi_projector::Point2D, 4>& quad,
                      const std::array<roi_projector::Point2D, 4>& barcode) {
  // 检查点是否有效
  for (const auto& pt : quad) {
    if (!std::isfinite(pt.u) || !std::isfinite(pt.v)) {
      return 0.0;
    }
  }
  for (const auto& pt : barcode) {
    if (!std::isfinite(pt.u) || !std::isfinite(pt.v)) {
      return 0.0;
    }
  }
  
  // 计算两个四边形的面积
  std::vector<roi_projector::Point2D> quad_vec(quad.begin(), quad.end());
  std::vector<roi_projector::Point2D> barcode_vec(barcode.begin(), barcode.end());
  
  const double area_quad = ComputePolygonArea(quad_vec);
  const double area_barcode = ComputePolygonArea(barcode_vec);
  
  if (area_quad < 1e-9 || area_barcode < 1e-9) {
    return 0.0;
  }
  
  // 计算交集面积
  // ComputeConvexPolygonIntersection(subject, clip) 返回 subject 在 clip 内部的部分
  // 所以 ComputeConvexPolygonIntersection(barcode, quad) 返回 barcode 在 quad 内的部分
  // 这就是我们需要的交集：barcode ∩ quad
  const auto intersection = ComputeConvexPolygonIntersection(barcode, quad);
  
  // 调试输出 - 使用cout并立即刷新
  std::cout << "[IOU Debug] area_quad=" << area_quad << ", area_barcode=" << area_barcode 
            << ", intersection_size=" << intersection.size();
  if (intersection.size() > 0) {
    std::cout << ", first_point=(" << intersection[0].u << "," << intersection[0].v << ")";
  }
  std::cout << std::endl << std::flush;
  
  const double intersection_area = ComputePolygonArea(intersection);
  
  // IOU定义：码区有多少在ROI里面 = 交集面积 / barcode面积
  if (area_barcode < 1e-9) {
    return 0.0;
  }
  
  return intersection_area / area_barcode;
}

}  // namespace

bool IsRoiInsideQuad(const std::array<Point2D, 4>& quad, const std::array<Point2D, 4>& barcode) {
  constexpr double kIOUThreshold = 0.8;
  const double iou = ComputeIOUImpl(quad, barcode);
  // 调试输出：打印IOU值 - 使用cout并立即刷新
  std::cout << "[IOU Debug] IOU: " << iou << std::endl << std::flush;
  return iou > kIOUThreshold;
}

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

CornersResult Projector::ProjectCorners(
    const std::array<Point3D, 4>& corners) const {
  CornersResult result;
  if (!has_calibration_) {
    result.message = "calibration not loaded";
    return result;
  }

  for (size_t i = 0; i < corners.size(); ++i) {
    const Point3D& pt = corners[i];
    if (pt.z <= 0.0 || !std::isfinite(pt.z)) {
      result.message = "invalid depth at corner " + std::to_string(i);
      return result;
    }
    double out_u = 0.0;
    double out_v = 0.0;
    if (!TransformPoint(pt.u, pt.v, pt.z, out_u, out_v)) {
      result.message = "projection failed at corner " + std::to_string(i);
      return result;
    }
    result.points[i].u = out_u;
    result.points[i].v = out_v;
  }

  result.ok = true;
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

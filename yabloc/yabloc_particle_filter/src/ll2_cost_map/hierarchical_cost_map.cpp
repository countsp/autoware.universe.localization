// Copyright 2023 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//生成和管理一个分层的成本地图。这个成本地图基于来自传感器的数据，例如激光雷达扫描或视觉识别，以帮助自动驾驶系统或机器人进行路径规划和避障。


#include "yabloc_particle_filter/ll2_cost_map/hierarchical_cost_map.hpp"

#include "yabloc_particle_filter/ll2_cost_map/direct_cost_map.hpp"

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <yabloc_common/color.hpp>

#include <boost/geometry/geometry.hpp>

namespace yabloc
{
float Area::unit_length_ = -1;

HierarchicalCostMap::HierarchicalCostMap(rclcpp::Node * node)
: max_range_(node->declare_parameter<float>("max_range")),
  image_size_(node->declare_parameter<int>("image_size")),
  max_map_count_(10),
  logger_(node->get_logger())
{
  Area::unit_length_ = max_range_;
  float gamma = node->declare_parameter<float>("gamma");
  gamma_converter.reset(gamma);
}

//将一个给定位置转换为成本地图上的对应像素位置。
cv::Point2i HierarchicalCostMap::to_cv_point(const Area & area, const Eigen::Vector2f p) const
{
  Eigen::Vector2f relative = p - area.real_scale();
  float px = relative.x() / max_range_ * image_size_;
  float py = relative.y() / max_range_ * image_size_;
  return {static_cast<int>(px), static_cast<int>(py)};
}

//根据给定的位置查询成本地图，返回该位置的成本值。
CostMapValue HierarchicalCostMap::at(const Eigen::Vector2f & position)
{
  if (!cloud_.has_value()) {
    return CostMapValue{0.5f, 0, true};
  }

  Area key(position);
  if (cost_maps_.count(key) == 0) {
    build_map(key);
  }
  map_accessed_[key] = true;

  cv::Point2i tmp = to_cv_point(key, position);
  cv::Vec3b b3 = cost_maps_.at(key).ptr<cv::Vec3b>(tmp.y)[tmp.x];
  return {b3[0] / 255.f, b3[1], b3[2] == 1};
}

//设置成本地图的高度，用于处理不同高度层的导航。
void HierarchicalCostMap::set_height(float height)
{
  if (height_) {
    if (std::abs(*height_ - height) > 2) {
      generated_map_history_.clear();
      cost_maps_.clear();
      map_accessed_.clear();
    }
  }

  height_ = height;
}

//设置障碍物的边界框，用于成本地图的生成。
void HierarchicalCostMap::set_bounding_box(const pcl::PointCloud<pcl::PointXYZL> & cloud)
{
  if (cloud.empty()) return;
  BgPolygon poly;

  std::optional<uint32_t> last_label = std::nullopt;
  for (const pcl::PointXYZL p : cloud) {
    if (last_label) {
      if ((*last_label) != p.label) {
        bounding_boxes_.push_back(poly);
        poly.outer().clear();
      }
    }
    poly.outer().push_back(BgPoint(p.x, p.y));
    last_label = p.label;
  }
  bounding_boxes_.push_back(poly);
}

//设置用于生成成本地图的点云数据。
void HierarchicalCostMap::set_cloud(const pcl::PointCloud<pcl::PointNormal> & cloud)
{
  cloud_ = cloud;
}

//为指定区域生成成本地图
void HierarchicalCostMap::build_map(const Area & area)
{
  if (!cloud_.has_value()) return;

  cv::Mat image = 255 * cv::Mat::ones(cv::Size(image_size_, image_size_), CV_8UC1);
  cv::Mat orientation = cv::Mat::zeros(cv::Size(image_size_, image_size_), CV_8UC1);

  auto cvPoint = [this, area](const Eigen::Vector3f & p) -> cv::Point {
    return this->to_cv_point(area, p.topRows(2));
  };

  // TODO(KYabuuchi) We can speed up by skipping too far line_segments
  for (const auto pn : cloud_.value()) {
    if (height_) {
      if (std::abs(pn.z - *height_) > 4) continue;
      if (std::abs(pn.normal_z - *height_) > 4) continue;
    }

    cv::Point2i from = cvPoint(pn.getVector3fMap());
    cv::Point2i to = cvPoint(pn.getNormalVector3fMap());

    float radian = std::atan2(from.y - to.y, from.x - to.x);
    if (radian < 0) radian += M_PI;
    float degree = radian * 180 / M_PI;

    cv::line(image, from, to, cv::Scalar::all(0), 1);
    cv::line(orientation, from, to, cv::Scalar::all(degree), 1);
  }

  // channel-1
  cv::Mat distance;
  cv::distanceTransform(image, distance, cv::DIST_L2, 3);
  cv::threshold(distance, distance, 100, 100, cv::THRESH_TRUNC);
  distance.convertTo(distance, CV_8UC1, -2.55, 255);

  // channel-2
  cv::Mat whole_orientation = direct_cost_map(orientation, image);

  // channel-3
  cv::Mat available_area = create_available_area_image(area);

  cv::Mat directed_cost_map;
  cv::merge(
    std::vector<cv::Mat>{gamma_converter(distance), whole_orientation, available_area},
    directed_cost_map);

  cost_maps_[area] = directed_cost_map;
  generated_map_history_.push_back(area);

  RCLCPP_INFO_STREAM(
    logger_, "succeeded to build map " << area(area) << " " << area.real_scale().transpose());
}

//生成一个可视化标记数组，显示生成的成本地图的范围。
HierarchicalCostMap::MarkerArray HierarchicalCostMap::show_map_range() const
{
  MarkerArray array_msg;

  auto point_msg = [](float x, float y) -> geometry_msgs::msg::Point {
    geometry_msgs::msg::Point gp;
    gp.x = x;
    gp.y = y;
    return gp;
  };

  int id = 0;
  for (const Area & area : generated_map_history_) {
    Marker marker;
    marker.header.frame_id = "map";
    marker.id = id++;
    marker.type = Marker::LINE_STRIP;
    marker.color = common::Color(0, 0, 1.0f, 1.0f);
    marker.scale.x = 0.1;
    Eigen::Vector2f xy = area.real_scale();
    marker.points.push_back(point_msg(xy.x(), xy.y()));
    marker.points.push_back(point_msg(xy.x() + area.unit_length_, xy.y()));
    marker.points.push_back(point_msg(xy.x() + area.unit_length_, xy.y() + area.unit_length_));
    marker.points.push_back(point_msg(xy.x(), xy.y() + area.unit_length_));
    marker.points.push_back(point_msg(xy.x(), xy.y()));
    array_msg.markers.push_back(marker);
  }
  return array_msg;
}

//根据当前位置生成成本地图的图像。
cv::Mat HierarchicalCostMap::get_map_image(const Pose & pose)
{
  // if (generated_map_history_.empty())
  //   return cv::Mat::zeros(cv::Size(image_size_, image_size_), CV_8UC3);

  Eigen::Vector2f center;
  center << pose.position.x, pose.position.y;

  float w = pose.orientation.w;
  float z = pose.orientation.z;
  Eigen::Matrix2f R = Eigen::Rotation2Df(2.f * std::atan2(z, w) - M_PI_2).toRotationMatrix();

  auto toVector2f = [this, center, R](float h, float w) -> Eigen::Vector2f {
    Eigen::Vector2f offset;
    offset.x() = (w / this->image_size_ - 0.5f) * this->max_range_ * 1.5f;
    offset.y() = -(h / this->image_size_ - 0.5f) * this->max_range_ * 1.5f;
    return center + R * offset;
  };

  cv::Mat image = cv::Mat::zeros(cv::Size(image_size_, image_size_), CV_8UC3);
  for (int w = 0; w < image_size_; w++) {
    for (int h = 0; h < image_size_; h++) {
      CostMapValue v3 = this->at(toVector2f(h, w));
      if (v3.unmapped)
        image.at<cv::Vec3b>(h, w) = cv::Vec3b(v3.angle, 255 * v3.intensity, 50);
      else
        image.at<cv::Vec3b>(h, w) = cv::Vec3b(v3.angle, 255 * v3.intensity, 255 * v3.intensity);
    }
  }

  cv::Mat rgb_image;
  cv::cvtColor(image, rgb_image, cv::COLOR_HSV2BGR);
  return rgb_image;
}

//删除不再需要的成本地图数据，以节省内存。
void HierarchicalCostMap::erase_obsolete()
{
  if (cost_maps_.size() < max_map_count_) return;

  for (auto itr = generated_map_history_.begin(); itr != generated_map_history_.end();) {
    if (map_accessed_[*itr]) {
      ++itr;
      continue;
    }
    cost_maps_.erase(*itr);
    itr = generated_map_history_.erase(itr);
  }

  map_accessed_.clear();
}

//创建表示可导航和不可导航区域的图像。
cv::Mat HierarchicalCostMap::create_available_area_image(const Area & area) const
{
  cv::Mat available_area = cv::Mat::zeros(cv::Size(image_size_, image_size_), CV_8UC1);
  if (bounding_boxes_.empty()) return available_area;

  // Define current area
  using BgBox = boost::geometry::model::box<BgPoint>;

  BgBox area_polygon;
  std::array<Eigen::Vector2f, 2> area_bounding_box = area.real_scale_boundary();
  area_polygon.min_corner() = {area_bounding_box[0].x(), area_bounding_box[0].y()};
  area_polygon.max_corner() = {area_bounding_box[1].x(), area_bounding_box[1].y()};

  std::vector<std::vector<cv::Point2i>> contours;

  for (const BgPolygon & box : bounding_boxes_) {
    if (boost::geometry::disjoint(area_polygon, box)) {
      continue;
    }
    std::vector<cv::Point2i> contour;
    for (BgPoint p : box.outer()) {
      contour.push_back(to_cv_point(area, {p.x(), p.y()}));
    }
    contours.push_back(contour);
  }

  cv::drawContours(available_area, contours, -1, cv::Scalar::all(1), -1);
  return available_area;
}

}  // namespace yabloc

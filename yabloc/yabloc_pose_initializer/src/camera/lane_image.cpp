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
//利用Lanelet地图数据，通过将车道线和车道多边形绘制到图像上，为自动驾驶车辆提供了一种直观的车道信息表示方式。
#include "yabloc_pose_initializer/camera/lane_image.hpp"

#include <opencv2/imgproc.hpp>

#include <boost/assert.hpp>
#include <boost/geometry/algorithms/disjoint.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/range/adaptors.hpp>

namespace yabloc
{
namespace bg = boost::geometry;

typedef bg::model::d2::point_xy<double> point_t;
typedef bg::model::box<point_t> box_t;
typedef bg::model::polygon<point_t> polygon_t;

LaneImage::LaneImage(lanelet::LaneletMapPtr map) : map_(map)
{
}

//将给定的三维点转换为OpenCV图像坐标系中的二维点。
cv::Point2i to_cv_point(const Eigen::Vector3f & v)
{
  const float image_size_ = 800;
  const float max_range_ = 30;

  cv::Point pt;
  pt.x = -v.y() / max_range_ * image_size_ * 0.5f + image_size_ / 2.f;
  pt.y = -v.x() / max_range_ * image_size_ * 0.5f + image_size_ / 2.f;
  return pt;
}

// 根据指定位置生成车道信息的图像。
cv::Mat LaneImage::create_vector_map_image(const Eigen::Vector3f & position)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = position.x();
  pose.position.y = position.y();
  pose.position.z = position.z();
  return get_image(pose);
}

// 在图像上绘制一个车道多边形
void draw_lane(cv::Mat & image, const polygon_t & polygon)
{
  std::vector<cv::Point> contour;
  for (auto p : polygon.outer()) {
    cv::Point2i pt = to_cv_point(Eigen::Vector3f(p.x(), p.y(), 0));
    contour.push_back(pt);
  }

  std::vector<std::vector<cv::Point> > contours;
  contours.push_back(contour);
  cv::drawContours(image, contours, -1, cv::Scalar(255, 0, 0), -1);
}

// 在图像上绘制一个车道线。
void draw_line(cv::Mat & image, const lanelet::LineString2d & line, geometry_msgs::msg::Point xyz)
{
  std::vector<cv::Point> contour;
  for (auto p : line) {
    cv::Point2i pt = to_cv_point(Eigen::Vector3f(p.x() - xyz.x, p.y() - xyz.y, 0));
    contour.push_back(pt);
  }
  cv::polylines(image, contour, false, cv::Scalar(0, 0, 255), 2);
}

//根据给定的位姿生成表示车道信息的图像。
cv::Mat LaneImage::get_image(const Pose & pose)
{
  const auto xyz = pose.position;
  box_t box(point_t(-20, -20), point_t(20, 20));

  cv::Mat image = cv::Mat::zeros(cv::Size(800, 800), CV_8UC3);

  std::vector<lanelet::Lanelet> joint_lanes;
  for (auto lanelet : map_->laneletLayer) {
    polygon_t polygon;
    for (auto right : lanelet.rightBound2d()) {
      polygon.outer().push_back(point_t(right.x() - xyz.x, right.y() - xyz.y));
    }
    for (auto left : boost::adaptors::reverse(lanelet.leftBound2d())) {
      polygon.outer().push_back(point_t(left.x() - xyz.x, left.y() - xyz.y));
    }

    if (!bg::disjoint(box, polygon)) {
      joint_lanes.push_back(lanelet);
      draw_lane(image, polygon);
    }
  }
  for (auto lanelet : joint_lanes) {
    draw_line(image, lanelet.rightBound2d(), xyz);
    draw_line(image, lanelet.leftBound2d(), xyz);
  }

  return image;
}

}  // namespace yabloc

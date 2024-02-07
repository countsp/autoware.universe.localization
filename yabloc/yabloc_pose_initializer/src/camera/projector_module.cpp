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
//桥接摄像机捕获的图像数据和车辆的物理世界，使得可以在图像处理结果上执行几何变换，从而在物理世界中定位和解释这些结果。

#include "yabloc_pose_initializer/camera/projector_module.hpp"

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>
#include <yabloc_common/cv_decompress.hpp>

namespace yabloc::initializer
{
ProjectorModule::ProjectorModule(rclcpp::Node * node)
: info_(node), tf_subscriber_(node->get_clock()), logger_(node->get_logger())
{
}
//将三维点转换为二维图像点，用于在图像上绘制投影后的点。
cv::Point2i to_cv_point(const Eigen::Vector3f & v)
{
  const float image_size_ = 800;
  const float max_range_ = 30;

  cv::Point pt;
  pt.x = -v.y() / max_range_ * image_size_ * 0.5f + image_size_ / 2.f;
  pt.y = -v.x() / max_range_ * image_size_ * 0.5f + image_size_ / 2.f;
  return pt;
}

//将分割掩码图像中的特征（如道路、车辆等）根据摄像机参数和摄像机到车辆基座的变换，投影到一个预定义平面（如车辆底盘平面）上。
cv::Mat ProjectorModule::project_image(const cv::Mat & mask_image)
{
  // project semantics on plane
  std::vector<cv::Mat> masks;
  cv::split(mask_image, masks);
  std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};

  cv::Mat projected_image = cv::Mat::zeros(cv::Size(800, 800), CV_8UC3);
  for (int i = 0; i < 3; i++) {
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(masks[i], contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    std::vector<std::vector<cv::Point> > projected_contours;
    for (auto contour : contours) {
      std::vector<cv::Point> projected;
      for (auto c : contour) {
        auto opt = project_func_(c);
        if (!opt.has_value()) continue;

        cv::Point2i pt = to_cv_point(opt.value());
        projected.push_back(pt);
      }
      if (projected.size() > 2) {
        projected_contours.push_back(projected);
      }
    }
    cv::drawContours(projected_image, projected_contours, -1, colors[i], -1);
  }
  return projected_image;
}

//定义一个闭包函数，用于将图像坐标点投影到二维平面上的具体位置。这需要摄像机的内参、外参以及地面平面的假设。
bool ProjectorModule::define_project_func()
{
  if (project_func_) return true;

  if (info_.is_camera_info_nullopt()) {
    RCLCPP_WARN_STREAM(logger_, "camera info is not ready");
    return false;
  }
  Eigen::Matrix3f intrinsic_inv = info_.intrinsic().inverse();

  std::optional<Eigen::Affine3f> camera_extrinsic =
    tf_subscriber_(info_.get_frame_id(), "base_link");
  if (!camera_extrinsic.has_value()) {
    RCLCPP_WARN_STREAM(logger_, "camera tf_static is not ready");
    return false;
  }

  const Eigen::Vector3f t = camera_extrinsic->translation();
  const Eigen::Quaternionf q(camera_extrinsic->rotation());

  // TODO(KYabuuchi) This will take into account ground tilt and camera vibration someday.
  project_func_ = [intrinsic_inv, q, t](const cv::Point & u) -> std::optional<Eigen::Vector3f> {
    Eigen::Vector3f u3(u.x, u.y, 1);
    Eigen::Vector3f u_bearing = (q * intrinsic_inv * u3).normalized();
    if (u_bearing.z() > -0.01) return std::nullopt;
    float u_distance = -t.z() / u_bearing.z();
    Eigen::Vector3f v;
    v.x() = t.x() + u_bearing.x() * u_distance;
    v.y() = t.y() + u_bearing.y() * u_distance;
    v.z() = 0;
    return v;
  };
  return true;
}
}  // namespace yabloc::initializer

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

#include "yabloc_common/static_tf_subscriber.hpp"

namespace yabloc::common
{
//订阅和获取静态坐标变换（TF）信息,并将其转换为Sophus库的 SE3f 结构或Eigen库的 Affine3f 结构。
StaticTfSubscriber::StaticTfSubscriber(rclcpp::Clock::SharedPtr clock)
{
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(clock);
  transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

//获取指定的坐标帧之间的变换，并将其转换为 Sophus::SE3f 结构。
std::optional<Sophus::SE3f> StaticTfSubscriber::se3f(
  const std::string & frame_id, const std::string & parent_frame_id)
{
  std::optional<Eigen::Affine3f> opt_aff = (*this)(frame_id, parent_frame_id);
  if (!opt_aff.has_value()) return std::nullopt;

  Sophus::SE3f se3f(opt_aff->rotation(), opt_aff->translation());
  return se3f;
}

//重载操作符，用于直接获取指定的坐标帧之间的变换，并将其转换为 Eigen::Affine3f 结构。
std::optional<Eigen::Affine3f> StaticTfSubscriber::operator()(
  const std::string & frame_id, const std::string & parent_frame_id)
{
  std::optional<Eigen::Affine3f> extrinsic_{std::nullopt};
  try {
    geometry_msgs::msg::TransformStamped ts =
      tf_buffer_->lookupTransform(parent_frame_id, frame_id, tf2::TimePointZero);
    Eigen::Vector3f p;
    p.x() = ts.transform.translation.x;
    p.y() = ts.transform.translation.y;
    p.z() = ts.transform.translation.z;

    Eigen::Quaternionf q;
    q.w() = ts.transform.rotation.w;
    q.x() = ts.transform.rotation.x;
    q.y() = ts.transform.rotation.y;
    q.z() = ts.transform.rotation.z;
    extrinsic_ = Eigen::Affine3f::Identity();
    extrinsic_->translation() = p;
    extrinsic_->matrix().topLeftCorner(3, 3) = q.toRotationMatrix();
  } catch (tf2::TransformException & ex) {
  }
  return extrinsic_;
}

}  // namespace yabloc::common

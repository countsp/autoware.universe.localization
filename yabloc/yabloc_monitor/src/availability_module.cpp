// Copyright 2023 TIER IV
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

//评估YabLoc的可用性。它是通过监听YabLoc的姿态信息（pose）并检查最新姿态消息的时间戳来实现的。

#include "availability_module.hpp"

#include <rclcpp/logging.hpp>

#include <memory>

AvailabilityModule::AvailabilityModule(rclcpp::Node * node)
: clock_(node->get_clock()),
  latest_yabloc_pose_stamp_(std::nullopt),
  timestamp_threshold_(node->declare_parameter<double>("availability/timestamp_tolerance"))
{
  sub_yabloc_pose_ = node->create_subscription<PoseStamped>(
    "~/input/yabloc_pose", 10,
    [this](const PoseStamped::ConstSharedPtr msg) { on_yabloc_pose(msg); });
}

//检查YabLoc是否可用。如果自上次收到姿态信息以来的时间小于设定的阈值，则认为YabLoc可用。
bool AvailabilityModule::is_available() const
{
  if (!latest_yabloc_pose_stamp_.has_value()) {
    return false;
  }

  const auto now = clock_->now();

  const auto diff_time = now - latest_yabloc_pose_stamp_.value();
  const auto diff_time_sec = diff_time.seconds();
  return diff_time_sec < timestamp_threshold_;
}

//姿态信息的回调函数。更新最新姿态消息的时间戳。
void AvailabilityModule::on_yabloc_pose(const PoseStamped::ConstSharedPtr msg)
{
  latest_yabloc_pose_stamp_ = rclcpp::Time(msg->header.stamp);
}

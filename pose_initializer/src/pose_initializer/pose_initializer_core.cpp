// Copyright 2022 The Autoware Contributors
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

#include "pose_initializer_core.hpp"

#include "copy_vector_to_array.hpp"
#include "ekf_localization_trigger_module.hpp"
#include "gnss_module.hpp"
#include "ndt_localization_trigger_module.hpp"
#include "ndt_module.hpp"
#include "stop_check_module.hpp"
#include "yabloc_module.hpp"

#include <memory>
#include <vector>

//配置各种定位模块（GNSS、NDT等）的可用性，并创建相应的服务和发布者
PoseInitializer::PoseInitializer() : Node("pose_initializer")
{
  const auto node = component_interface_utils::NodeAdaptor(this);
  group_srv_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  node.init_pub(pub_state_);
  node.init_srv(srv_initialize_, this, &PoseInitializer::on_initialize, group_srv_);
  pub_reset_ = create_publisher<PoseWithCovarianceStamped>("pose_reset", 1);

  output_pose_covariance_ = get_covariance_parameter(this, "output_pose_covariance");
  gnss_particle_covariance_ = get_covariance_parameter(this, "gnss_particle_covariance");

  if (declare_parameter<bool>("ekf_enabled")) {
    ekf_localization_trigger_ = std::make_unique<EkfLocalizationTriggerModule>(this);
  }
  if (declare_parameter<bool>("gnss_enabled")) {
    gnss_ = std::make_unique<GnssModule>(this);
  }
  if (declare_parameter<bool>("yabloc_enabled")) {
    yabloc_ = std::make_unique<YabLocModule>(this);
  }
  if (declare_parameter<bool>("ndt_enabled")) {
    ndt_ = std::make_unique<NdtModule>(this);
    ndt_localization_trigger_ = std::make_unique<NdtLocalizationTriggerModule>(this);
  }
  if (declare_parameter<bool>("stop_check_enabled")) {
    // Add 1.0 sec margin for twist buffer.
    stop_check_duration_ = declare_parameter<double>("stop_check_duration");
    stop_check_ = std::make_unique<StopCheckModule>(this, stop_check_duration_ + 1.0);
  }
  logger_configure_ = std::make_unique<tier4_autoware_utils::LoggerLevelConfigure>(this);

  change_state(State::Message::UNINITIALIZED);
}

PoseInitializer::~PoseInitializer()
{
  // to delete gnss module
}

//更改并发布当前初始化器的状态
void PoseInitializer::change_state(State::Message::_state_type state)
{
  state_.stamp = now();
  state_.state = state;
  pub_state_->publish(state_);
}

//    req:包含期望的初始位姿。res:初始化是否成功
//    检查是否启用了停车检测，如果车辆未停止，则抛出异常。
//    如果配置了EKF或NDT触发模块，则先禁用它们以避免在初始化过程中更新位姿。
//    获取GNSS位姿，如果请求中包含了位姿，则使用请求的位姿。
//    如果启用了NDT或YabLoc，则尝试使用这些方法对位姿进行进一步对齐和优化。
//    发布重置后的位姿，并根据配置重新启动EKF和NDT。
//    更新和发布当前的状态。
void PoseInitializer::on_initialize(
  const Initialize::Service::Request::SharedPtr req,
  const Initialize::Service::Response::SharedPtr res)
{
  // NOTE: This function is not executed during initialization because mutually exclusive.
  if (stop_check_ && !stop_check_->isVehicleStopped(stop_check_duration_)) {
    throw ServiceException(
      Initialize::Service::Response::ERROR_UNSAFE, "The vehicle is not stopped.");
  }
  try {
    change_state(State::Message::INITIALIZING);
    if (ekf_localization_trigger_) {
      ekf_localization_trigger_->send_request(false);
    }
    if (ndt_localization_trigger_) {
      ndt_localization_trigger_->send_request(false);
    }
    auto pose = req->pose.empty() ? get_gnss_pose() : req->pose.front();
    if (ndt_) {
      pose = ndt_->align_pose(pose);
    } else if (yabloc_) {
      // If both the NDT and YabLoc initializer are enabled, prioritize NDT as it offers more
      // accuracy pose.
      pose = yabloc_->align_pose(pose);
    }
    pose.pose.covariance = output_pose_covariance_;
    pub_reset_->publish(pose);
    if (ekf_localization_trigger_) {
      ekf_localization_trigger_->send_request(true);
    }
    if (ndt_localization_trigger_) {
      ndt_localization_trigger_->send_request(true);
    }
    res->status.success = true;
    change_state(State::Message::INITIALIZED);
  } catch (const ServiceException & error) {
    res->status = error.status();
    change_state(State::Message::UNINITIALIZED);
  }
}

//如果GNSS模块可用，则从该模块获取当前位姿，并应用预定义的粒子协方差。如果GNSS不支持，则抛出异常。
geometry_msgs::msg::PoseWithCovarianceStamped PoseInitializer::get_gnss_pose()
{
  if (gnss_) {
    PoseWithCovarianceStamped pose = gnss_->get_pose();
    pose.pose.covariance = gnss_particle_covariance_;
    return pose;
  }
  throw ServiceException(
    Initialize::Service::Response::ERROR_GNSS_SUPPORT, "GNSS is not supported.");
}

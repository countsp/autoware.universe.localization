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

//定期检查YabLoc的可用性状态，并通过ROS的诊断框架发布状态，使得系统状态可以通过如 rqt_robot_monitor 等工具进行实时监控和可视化。
#include "yabloc_monitor_core.hpp"

#include <rclcpp/logging.hpp>

#include <memory>

YabLocMonitor::YabLocMonitor() : Node("yabloc_monitor"), updater_(this)
{
  updater_.setHardwareID(get_name());
  updater_.add("yabloc_status", this, &YabLocMonitor::update_diagnostics);

  // Set timer
  using std::chrono_literals::operator""ms;
  timer_ = create_wall_timer(100ms, [this] { on_timer(); });
  // Evaluation modules
  availability_module_ = std::make_unique<AvailabilityModule>(this);
}

//定时器的回调函数，用于强制更新诊断信息。
void YabLocMonitor::on_timer()
{
  updater_.force_update();
}

//更新YabLoc状态的诊断信息。根据 AvailabilityModule 的结果，设置诊断状态和消息。
void YabLocMonitor::update_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  bool is_available = availability_module_->is_available();
  stat.add("Availability", is_available ? "OK" : "NG");

  if (is_available) {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "OK");
  } else {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "NG");
  }
}

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
//用于粒子滤波器校正步骤的抽象基类。它提供了基础功能来处理粒子数组的订阅和发布，同时支持可选的可视化功能。

#include "yabloc_particle_filter/correction/abstract_corrector.hpp"

namespace yabloc::modularized_particle_filter
{
AbstractCorrector::AbstractCorrector(const std::string & node_name)
: Node(node_name),
  acceptable_max_delay_(declare_parameter<float>("acceptable_max_delay")),
  visualize_(declare_parameter<bool>("visualize")),
  logger_(rclcpp::get_logger("abstract_corrector"))
{
  using std::placeholders::_1;
  particle_pub_ = create_publisher<ParticleArray>("~/output/weighted_particles", 10);
  particle_sub_ = create_subscription<ParticleArray>(
    "~/input/predicted_particles", 10, std::bind(&AbstractCorrector::on_particle_array, this, _1));

  if (visualize_) visualizer_ = std::make_shared<ParticleVisualizer>(*this);
}

// 订阅粒子数组消息的回调函数，将接收到的粒子数组添加到内部缓冲区中。
void AbstractCorrector::on_particle_array(const ParticleArray & particle_array)
{
  particle_array_buffer_.push_back(particle_array);
}

//从内部缓冲区中获取与给定时间戳同步的粒子数组，基于可接受的最大延迟。
std::optional<AbstractCorrector::ParticleArray> AbstractCorrector::get_synchronized_particle_array(
  const rclcpp::Time & stamp)
{
  auto itr = particle_array_buffer_.begin();
  while (itr != particle_array_buffer_.end()) {
    rclcpp::Duration dt = rclcpp::Time(itr->header.stamp) - stamp;
    if (dt.seconds() < -acceptable_max_delay_)
      particle_array_buffer_.erase(itr++);
    else
      break;
  }

  if (particle_array_buffer_.empty()) {
    RCLCPP_WARN_STREAM_THROTTLE(
      logger_, *get_clock(), 2000, "synchronized particles are requested but buffer is empty");
  }

  if (particle_array_buffer_.empty()) return std::nullopt;

  auto comp = [stamp](ParticleArray & x1, ParticleArray & x2) -> bool {
    auto dt1 = rclcpp::Time(x1.header.stamp) - stamp;
    auto dt2 = rclcpp::Time(x2.header.stamp) - stamp;
    return std::abs(dt1.seconds()) < std::abs(dt2.seconds());
  };
  return *std::min_element(particle_array_buffer_.begin(), particle_array_buffer_.end(), comp);
}

//发布加权粒子数组，并根据配置进行可视化。
void AbstractCorrector::set_weighted_particle_array(const ParticleArray & particle_array)
{
  particle_pub_->publish(particle_array);
  if (visualize_) visualizer_->publish(particle_array);
}

}  // namespace yabloc::modularized_particle_filter

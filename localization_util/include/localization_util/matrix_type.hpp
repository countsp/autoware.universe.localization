// Copyright 2021 TierIV
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

#ifndef LOCALIZATION_UTIL__MATRIX_TYPE_HPP_
#define LOCALIZATION_UTIL__MATRIX_TYPE_HPP_

#include <Eigen/Core>

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

#endif  // LOCALIZATION_UTIL__MATRIX_TYPE_HPP_

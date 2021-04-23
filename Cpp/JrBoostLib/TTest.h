//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


enum class TestDirection { Up, Down, Any };


ArrayXf tStatistic(
    Eigen::Ref<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> inData,
    CRefXs outData,
    optional<CRefXs> optSamples = optional<CRefXs>()
);


ArrayXs tTestRank(
    Eigen::Ref<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> inData,
    CRefXs outData,
    optional<CRefXs> optSamples = optional<CRefXs>(),
    TestDirection testDirection = TestDirection::Any
);

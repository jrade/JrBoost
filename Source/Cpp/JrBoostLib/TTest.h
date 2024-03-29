//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


enum class TestDirection { Up, Down, Any };


ArrayXf tStatistic(CRefXXfr inData, CRefXu8 outData, optional<CRefXs> samples = std::nullopt);

ArrayXs tTestRank(
    CRefXXfr inData, CRefXu8 outData, optional<CRefXs> samples = std::nullopt,
    TestDirection testDirection = TestDirection::Any);

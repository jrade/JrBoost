//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


ArrayXf fStatistic(CRefXXfr inData, CRefXu8 outData, optional<vector<size_t>> optSamples = std::nullopt);

ArrayXs fTestRank(CRefXXfr inData, CRefXu8 outData, optional<vector<size_t>> optSamples = std::nullopt);

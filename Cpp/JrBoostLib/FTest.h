//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


ArrayXf fStatistic(CRefXXfr inData, CRefXs outData, optional<vector<size_t>> optSamples = {});

ArrayXs fTestRank(CRefXXfr inData, CRefXs outData, optional<vector<size_t>> optSamples = {});

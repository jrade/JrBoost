//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

tuple<ArrayXs, ArrayXs, ArrayXd>
topScoringPairs(CRefXXfr inData, CRefXu8 outData, size_t pairCount, optional<CRefXs> samples = std::nullopt);

tuple<ArrayXs, ArrayXs, ArrayXd>
filterPairs(CRefXs variables1, CRefXs variables2, CRefXd scores, size_t singleVariableMaxFrequency);

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class BoostOptions;

using BoostParam = map<string, variant<bool, size_t, double>>;


BoostParam toBoostParam(const BoostOptions& opt);

BoostOptions toBoostOptions(const BoostParam& param);

vector<BoostOptions> toBoostOptions(const vector<BoostParam>& paramList);

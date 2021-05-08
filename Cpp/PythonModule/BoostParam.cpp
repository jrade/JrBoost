//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostParam.h"
#include "../JrBoostLib/BoostOptions.h"

namespace py = pybind11;


BoostParam toBoostParam(const BoostOptions& opt)
{
    BoostParam param;

    param["gamma"] = opt.gamma();
    param["iterationCount"] = opt.iterationCount();
    param["eta"] = opt.eta();
    param["fastExp"] = opt.fastExp();
    param["usedSampleRatio"] = opt.usedSampleRatio();
    param["usedVariableRatio"] = opt.usedVariableRatio();
    param["topVariableCount"] = opt.topVariableCount();
    param["minAbsSampleWeight"] = opt.minAbsSampleWeight();
    param["minRelSampleWeight"] = opt.minRelSampleWeight();
    param["minNodeSize"] = opt.minNodeSize();
    param["minNodeWeight"] = opt.minNodeWeight();
    param["isStratified"] = opt.isStratified();

    return param;
}

BoostOptions toBoostOptions(const BoostParam& param)
{
    BoostOptions opt;

    for (const auto& [key, value] : param) {
        try {
            if (key == "gamma")
                opt.setGamma(std::get<double>(value));
            else if (key == "iterationCount")
                opt.setIterationCount(std::get<size_t>(value));
            else if (key == "eta")
                opt.setEta(std::get<double>(value));
            else if (key == "fastExp")
                opt.setFastExp(std::get<bool>(value));
            else if (key == "usedSampleRatio")
                opt.setUsedSampleRatio(std::get<double>(value));
            else if (key == "usedVariableRatio")
                opt.setUsedVariableRatio(std::get<double>(value));
            else if (key == "topVariableCount")
                opt.setTopVariableCount(std::get<size_t>(value));
            else if (key == "minAbsSampleWeight")
                opt.setMinAbsSampleWeight(std::get<double>(value));
            else if (key == "minRelSampleWeight")
                opt.setMinRelSampleWeight(std::get<double>(value));
            else if (key == "minNodeSize")
                opt.setMinNodeSize(std::get<size_t>(value));
            else if (key == "minNodeWeight")
                opt.setMinNodeWeight(std::get<double>(value));
            else if (key == "isStratified")
                opt.setIsStratified(std::get<bool>(value));
            else {
                string msg = "The key '" + key + "' is not valid.";
                throw std::invalid_argument(msg);
            }
        }
        catch (const std::bad_variant_access&) {
            string msg = "The key '" + key + "' has value of wrong type.";
            throw std::invalid_argument(msg);
        }
    }

    return opt;
}

vector<BoostOptions> toBoostOptions(const vector<BoostParam>& paramList)
{
    vector<BoostOptions> optionsVector;
    optionsVector.reserve(paramList.size());
    for (const auto& param : paramList)
        optionsVector.push_back(toBoostOptions(param));
    return optionsVector;
}

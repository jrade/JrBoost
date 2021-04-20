//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostPredictor.h"
#include "BasePredictor.h"


BoostPredictor::~BoostPredictor() = default;


BoostPredictor::BoostPredictor(
    size_t variableCount,
    double c0,
    vector<double>&& c1,
    vector<unique_ptr<BasePredictor>>&& basePredictors
) :
    variableCount_(variableCount),
    c0_{ c0 },
    c1_{ std::move(c1) },
    basePredictors_{ std::move(basePredictors) }
{
    ASSERT(c1_.size() == basePredictors_.size());
}


ArrayXd BoostPredictor::predict(CRefXXf inData) const
{
    PROFILE::PUSH(PROFILE::BOOST_PREDICT);

    validateInData(inData);
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Constant(sampleCount, c0_);
    size_t n = basePredictors_.size();
    for (size_t k = 0; k < n; ++k)
        basePredictors_[k]->predict(inData, c1_[k], pred);
    pred = 1.0 / (1.0 + (-pred).exp());
    PROFILE::POP(sampleCount * n);
    return pred;
}



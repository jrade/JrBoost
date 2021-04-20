//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Predictor.h"


class EnsemblePredictor : public Predictor {
public:
    EnsemblePredictor(const vector<shared_ptr<Predictor>>& predictors) :
        predictors_(predictors)
    {
        ASSERT(!predictors_.empty());
        for (const auto& pred : predictors_)
            ASSERT(pred->variableCount() == variableCount());
    }

    virtual ~EnsemblePredictor() = default;

    virtual size_t variableCount() const { return predictors_[0]->variableCount(); }

private:
    virtual ArrayXd predictImpl_(CRefXXf inData) const
    {
        validateInData(inData);
        size_t sampleCount = static_cast<size_t>(inData.rows());
        ArrayXd pred = ArrayXd::Zero(sampleCount);
        size_t n = predictors_.size();
        for (size_t k = 0; k < n; ++k)
            pred += predictors_[k]->predictImpl_(inData);
        pred /= static_cast<double>(n);
        return pred;
    }

    vector<shared_ptr<Predictor>> predictors_;
};

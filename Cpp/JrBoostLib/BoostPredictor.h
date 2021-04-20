//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Predictor.h"

class BasePredictor;


class BoostPredictor : public Predictor {
public:
    virtual ~BoostPredictor();

    size_t variableCount() const { return variableCount_; }

private:
    friend class BoostTrainer;

    BoostPredictor(
        size_t variableCount,
        double c0,
        vector<double>&& c1,
        vector<unique_ptr<BasePredictor>>&& basePredictors
    );

    virtual ArrayXd predictImpl_(CRefXXf inData) const;

    size_t variableCount_;
    double c0_;
    vector<double> c1_;
    vector<unique_ptr<BasePredictor>> basePredictors_;
};

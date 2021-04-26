//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Predictor.h"

class BasePredictor;


class BoostPredictor : public Predictor {
public:
    BoostPredictor(
        size_t variableCount,
        double c0,
        double c1,
        vector<unique_ptr<BasePredictor>>&& basePredictors
    );
    virtual ~BoostPredictor();
    virtual size_t variableCount() const { return variableCount_; }

private:
    virtual ArrayXd predict_(CRefXXf inData) const;
    virtual void save_(ostream& os) const;

    friend class Predictor;
    static shared_ptr<Predictor> load_(istream& is);

    size_t variableCount_;
    float c0_;
    float c1_;
    vector<unique_ptr<BasePredictor>> basePredictors_;
};

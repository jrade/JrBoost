//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Predictor.h"


class EnsemblePredictor : public Predictor
{
public:
    EnsemblePredictor(const vector<shared_ptr<Predictor>>& predictors);

    virtual ~EnsemblePredictor() = default;
    virtual size_t variableCount() const { return predictors_[0]->variableCount(); }

private:
    virtual ArrayXd predict_(CRefXXf inData) const;
    virtual void variableWeights_(double c, RefXd weights) const;
    virtual void save_(ostream& os) const;

    friend class Predictor;
    static shared_ptr<Predictor> load_(istream& is, int version);

    vector<shared_ptr<Predictor>> predictors_;
};

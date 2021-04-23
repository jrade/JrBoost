//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Predictor.h"


class EnsemblePredictor : public Predictor {
public:
    EnsemblePredictor(const vector<shared_ptr<Predictor>>& predictors);

    virtual ~EnsemblePredictor() = default;
    virtual size_t variableCount() const { return predictors_[0]->variableCount(); }
    virtual void save(ostream& os) const;

private:
    virtual ArrayXd predictImpl_(CRefXXf inData) const;

    friend class Predictor;
    static shared_ptr<Predictor> loadImpl_(istream& is);

    vector<shared_ptr<Predictor>> predictors_;
};

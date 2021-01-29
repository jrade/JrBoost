#pragma once

#include "AbstractPredictor.h"


class LinearCombinationPredictor : public AbstractPredictor {
public:
    LinearCombinationPredictor(
        size_t variableCount,
        double c0,
        vector<double>&& c1,
        vector<unique_ptr<AbstractPredictor>>&& basePredictors
    );
    virtual ~LinearCombinationPredictor() = default;
    virtual ArrayXd predict(CRefXXf inData) const;

private:
    double c0_;
    vector<double> c1_;
    vector<unique_ptr<AbstractPredictor>> basePredictors_;
  };

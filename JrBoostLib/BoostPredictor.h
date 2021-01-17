#pragma once

#include "AbstractPredictor.h"

class BoostPredictor : public AbstractPredictor {
public:
    virtual ~BoostPredictor() = default;
    virtual size_t variableCount() const { return variableCount_; }
    virtual ArrayXd predict(CRefXXf inData) const;

private:
    BoostPredictor(size_t variableCount, double c0, vector<double>&& c1, vector<unique_ptr<AbstractPredictor>>&& basePredictors);

    friend class AdaBoostTrainer;
    friend class LogitBoostTrainer;

    size_t variableCount_;
    double c0_;
    vector<double> c1_;
    vector<unique_ptr<AbstractPredictor>> basePredictors_;
  };

#pragma once

#include "AbstractPredictor.h"

class BoostPredictor : public AbstractPredictor {
public:
    virtual ~BoostPredictor() = default;
    virtual size_t variableCount() const { return variableCount_; }
    virtual ArrayXf predict(RefXXf inData) const;

private:
    BoostPredictor(size_t variableCount, float f0, float eta, vector<unique_ptr<AbstractPredictor>>&& basePredictors);

    friend class AdaBoostTrainer;
    friend class LogitBoostTrainer;

    size_t variableCount_;
    float f0_;
    float eta_;
    vector<unique_ptr<AbstractPredictor>> basePredictors_;
  };

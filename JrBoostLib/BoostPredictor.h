#pragma once

#include "StumpPredictor.h"


class BoostPredictor {
public:
    BoostPredictor(BoostPredictor&&) = default;
    BoostPredictor& operator=(BoostPredictor&&) = default;
    ~BoostPredictor() = default;

    size_t variableCount() const { return variableCount_; }
    ArrayXd predict(CRefXXf inData) const;

// deleted:
    BoostPredictor() = delete;
    BoostPredictor(const BoostPredictor&) = delete;
    BoostPredictor& operator=(const BoostPredictor&) = delete;

private:
    BoostPredictor(size_t variableCount, double c0, vector<double>&& c1, vector<StumpPredictor>&& basePredictors);

    friend class AdaBoostTrainer;
    friend class LogitBoostTrainer;

    size_t variableCount_;
    double c0_;
    vector<double> c1_;
    vector<StumpPredictor> basePredictors_;
  };

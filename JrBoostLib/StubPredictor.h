#pragma once

#include "AbstractPredictor.h"

class StubPredictor : public AbstractPredictor {
public:
    virtual ~StubPredictor() = default;
    virtual int variableCount() const { return variableCount_; }
    virtual ArrayXf predict(const Eigen::ArrayXXf& inData) const;

private:
    StubPredictor(int variableCount) : variableCount_(variableCount) {}
    friend class StubTrainer;
    int variableCount_;
};

#pragma once

#include "AbstractPredictor.h"

class StubPredictor : public AbstractPredictor {
public:
    virtual ~StubPredictor() = default;
    virtual int variableCount() const { return variableCount_; }
    virtual ArrayXf predict(const Eigen::ArrayXXf& inData) const;

private:
    StubPredictor(int variableCount, int j, float x, float leftY, float rightY);
 
    friend class StubTrainer;

    int variableCount_;
    int j_;
    float x_;
    float leftY_;
    float rightY_;
};

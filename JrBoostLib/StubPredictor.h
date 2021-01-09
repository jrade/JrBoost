#pragma once

#include "AbstractPredictor.h"

class StubPredictor : public AbstractPredictor {
public:
    virtual ~StubPredictor() = default;
    virtual size_t variableCount() const { return variableCount_; }
    virtual ArrayXf predict(const Eigen::ArrayXXf& inData) const;

private:
    StubPredictor(size_t variableCount, size_t j, float x, float leftY, float rightY);
 
    friend class StubTrainer;

    size_t variableCount_;
    size_t j_;
    float x_;
    float leftY_;
    float rightY_;
};

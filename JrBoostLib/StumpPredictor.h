#pragma once

#include "AbstractPredictor.h"

class StumpPredictor : public AbstractPredictor {
public:
    virtual ~StumpPredictor() = default;
    virtual size_t variableCount() const { return variableCount_; }
    virtual ArrayXf predict(CRefXXf inData) const;

private:
    StumpPredictor(size_t variableCount, size_t j, float x, float leftY, float rightY);
 
    friend class StumpTrainer;

    size_t variableCount_;
    size_t j_;
    float x_;
    float leftY_;
    float rightY_;
};

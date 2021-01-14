#pragma once

#include "AbstractPredictor.h"

class TrivialPredictor : public AbstractPredictor {
public:
    TrivialPredictor(float value, size_t variableCount) :
        value_{ value },
        variableCount_{ variableCount }
    {}
    virtual ~TrivialPredictor() = default;
    virtual size_t variableCount() const { return variableCount_; }
    virtual ArrayXf predict(CRefXXf inData) const
    {
        ASSERT(static_cast<size_t>(inData.cols()) == variableCount_);
        size_t sampleCount = inData.rows();
        return ArrayXf::Constant(sampleCount, 1, value_);
    }

private:
    size_t variableCount_;
    float value_;
};

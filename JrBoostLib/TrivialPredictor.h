#pragma once

#include "AbstractPredictor.h"


class TrivialPredictor : public AbstractPredictor {
public:
    TrivialPredictor(size_t variableCount, double y) : 
        AbstractPredictor(variableCount),
        y_(y)
    {
        ASSERT(std::isfinite(y));
    }

    virtual ~TrivialPredictor() = default;

    virtual ArrayXd predict(CRefXXf inData) const
    {
        validateInData_(inData);
        const size_t sampleCount = inData.rows();
        return ArrayXd::Constant(sampleCount, y_);
    }

private:
    size_t variableCount_;
    double y_;
};

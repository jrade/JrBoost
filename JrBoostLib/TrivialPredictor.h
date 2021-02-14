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

protected:
    virtual void predictImpl_(CRefXXf inData, double c, RefXd outData) const
    {
        outData += c * y_;
    }

private:
    size_t variableCount_;
    double y_;
};

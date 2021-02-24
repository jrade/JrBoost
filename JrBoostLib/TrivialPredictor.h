#pragma once

#include "SimplePredictor.h"


class TrivialPredictor : public SimplePredictor {
public:
    TrivialPredictor(size_t variableCount, double y) : 
        SimplePredictor(variableCount),
        y_(y)
    {
        ASSERT(std::isfinite(y));
    }

    virtual ~TrivialPredictor() = default;

    virtual void predict(CRefXXf inData, double c, RefXd outData) const
    {
        outData += c * y_;
    }

private:
    double y_;
};

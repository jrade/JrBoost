#pragma once

#include "AbstractPredictor.h"


class StumpPredictor : public AbstractPredictor {
public:
    virtual ~StumpPredictor() = default;
    virtual ArrayXd predict(CRefXXf inData) const;
    virtual void predict(CRefXXf inData, double c, RefXd outData) const;

private:
    friend class StumpTrainer;

    StumpPredictor(size_t variableCount, size_t j, float x, double leftY, double rightY);

    size_t j_;
    float x_;
    double leftY_;
    double rightY_;
};

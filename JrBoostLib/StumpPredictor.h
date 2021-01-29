#pragma once

#include "AbstractPredictor.h"


class StumpPredictor : public AbstractPredictor {
public:
    virtual ~StumpPredictor() = default;
    virtual ArrayXd predict(CRefXXf inData) const;

private:
    friend class StumpTrainerByThread;

    StumpPredictor(size_t variableCount, size_t j, float x, double leftY, double rightY);

    size_t j_;
    float x_;
    double leftY_;
    double rightY_;
};

#pragma once

#include "AbstractPredictor.h"


class StumpPredictor : public AbstractPredictor {
public:
    virtual ~StumpPredictor() = default;

protected:
    virtual void predictImpl_(CRefXXf inData, double c, RefXd outData) const;

private:
    template<typename SampleIndex> friend class StumpTrainerImpl;

    StumpPredictor(size_t variableCount, size_t j, float x, double leftY, double rightY);

    size_t j_;
    float x_;
    double leftY_;
    double rightY_;
};

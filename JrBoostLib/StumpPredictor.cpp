#include "pch.h"
#include "StumpPredictor.h"

StumpPredictor::StumpPredictor(size_t variableCount, size_t j, float x, float leftY, float rightY) :
    variableCount_{ variableCount },
    j_{ j },
    x_{ x },
    leftY_{ leftY },
    rightY_{ rightY }
{}

ArrayXf StumpPredictor::predict(RefXXf inData) const
{
    size_t sampleCount = inData.rows();
    size_t variableCount = inData.cols();
    if (variableCount != variableCount_)
        throw std::runtime_error("The data does not have the same number of variables as the training data.");

    ArrayXf outData{ sampleCount };
    for (size_t i = 0; i < sampleCount; ++i)
        outData(i) = (inData(i, j_) < x_) ? leftY_ : rightY_;
    return outData;
}

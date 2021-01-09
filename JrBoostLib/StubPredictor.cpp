#include "pch.h"
#include "StubPredictor.h"

StubPredictor::StubPredictor(size_t variableCount, size_t j, float x, float leftY, float rightY) :
    variableCount_{ variableCount },
    j_{ j },
    x_{ x },
    leftY_{ leftY },
    rightY_{ rightY }
{}

ArrayXf StubPredictor::predict(const Eigen::ArrayXXf& inData) const
{
    if (static_cast<size_t>(inData.cols()) != variableCount_)
        throw std::runtime_error("The data does not have the same number of variables as the training data.");

    size_t sampleCount = inData.rows();
    ArrayXf outData(sampleCount);
    for (size_t i = 0; i < sampleCount; ++i)
        outData(i) = (inData(i, j_) < x_) ? leftY_ : rightY_;
    return outData;
}

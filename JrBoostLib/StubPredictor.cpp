#include "pch.h"
#include "StubPredictor.h"

StubPredictor::StubPredictor(int variableCount, int j, float x, float leftY, float rightY) :
    variableCount_(variableCount),
    j_(j),
    x_(x),
    leftY_(leftY),
    rightY_(rightY)
{}

ArrayXf StubPredictor::predict(const Eigen::ArrayXXf& inData) const
{
    if (inData.cols() != variableCount_)
        throw std::runtime_error("The data does not have the same number of variables as the training data.");

    int sampleCount = static_cast<int>(inData.rows());
    ArrayXf outData(sampleCount);
    for (int i = 0; i < sampleCount; ++i)
        outData(i) = (inData(i, j_) < x_) ? leftY_ : rightY_;
    return outData;
}

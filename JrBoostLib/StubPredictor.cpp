#include "pch.h"
#include "StubPredictor.h"

ArrayXf StubPredictor::predict(const Eigen::ArrayXXf& inData) const
{
    if (inData.cols() != variableCount_)
        throw std::runtime_error("The data does not have the same number of variables as the training data.");
    int sampleCount = static_cast<int>(inData.rows());
    return ArrayXf(sampleCount);
}

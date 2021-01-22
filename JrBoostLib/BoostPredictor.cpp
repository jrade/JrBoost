#include "pch.h"
#include "BoostPredictor.h"


BoostPredictor::BoostPredictor(
    size_t variableCount, double c0, vector<double>&& c1, vector<StumpPredictor>&& basePredictors
) :
    variableCount_{ variableCount },
    c0_{ c0 },
    c1_{ std::move(c1) },
    basePredictors_{ std::move(basePredictors) }
{}

ArrayXd BoostPredictor::predict(CRefXXf inData) const
{
    size_t sampleCount = inData.rows();
    size_t variableCount = inData.cols();

    if (variableCount != variableCount_)
        throw runtime_error("The data does not have the same number of variables as the training data.");

    ArrayXd outData{ Eigen::ArrayXd::Constant(sampleCount, c0_) };
    size_t n = basePredictors_.size();
    for (size_t k = 0; k < n; ++k)
        outData += c1_[k] * basePredictors_[k].predict(inData);
    return outData;
}


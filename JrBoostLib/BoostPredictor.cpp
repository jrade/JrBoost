#include "pch.h"
#include "BoostPredictor.h"

BoostPredictor::BoostPredictor(
    size_t variableCount, float f0, float eta, vector<unique_ptr<AbstractPredictor>>&& basePredictors
) :
    variableCount_(variableCount),
    f0_(f0),
    eta_(eta),
    basePredictors_(std::move(basePredictors))
{}

ArrayXf BoostPredictor::predict(RefXXf inData) const
{
    size_t sampleCount = inData.rows();
    size_t variableCount = inData.cols();

    if (variableCount != variableCount_)
        throw std::runtime_error("The data does not have the same number of variables as the training data.");

    ArrayXd outData{ ArrayXd::Constant(sampleCount, f0_) };
    for (auto& pred : basePredictors_)
        outData += eta_ * pred->predict(inData).cast<double>();
    return outData.cast<float>();
}

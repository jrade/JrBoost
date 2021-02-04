#include "pch.h"
#include "LinearCombinationPredictor.h"


LinearCombinationPredictor::LinearCombinationPredictor(
    size_t variableCount,
    double c0,
    vector<double>&& c1,
    vector<unique_ptr<AbstractPredictor>>&& basePredictors
) :
    AbstractPredictor(variableCount),
    c0_{ c0 },
    c1_{ std::move(c1) },
    basePredictors_{ std::move(basePredictors) }
{
    ASSERT(c1_.size() == basePredictors_.size());
}

ArrayXd LinearCombinationPredictor::predict(CRefXXf inData) const
{
    CLOCK::PUSH(CLOCK::LCP_P);

    validateInData_(inData);
    size_t sampleCount = inData.rows();
    ArrayXd outData{ Eigen::ArrayXd::Constant(sampleCount, c0_) };
    size_t n = basePredictors_.size();
    for (size_t k = 0; k < n; ++k)
        outData += c1_[k] * basePredictors_[k]->predict(inData);

    CLOCK::POP();

    return outData;
}

void LinearCombinationPredictor::predict(CRefXXf inData, double c, RefXd outData) const
{
    CLOCK::PUSH(CLOCK::LCP_P);

    validateInData_(inData);
    size_t n = basePredictors_.size();
    outData += c * c0_;
    for (size_t k = 0; k < n; ++k)
        basePredictors_[k]->predict(inData, c * c1_[k], outData);

    CLOCK::POP();
}

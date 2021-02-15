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

void LinearCombinationPredictor::predictImpl_(CRefXXf inData, double c, RefXd outData) const
{
    PROFILE::PUSH(PROFILE::LCP_P);

    size_t n = basePredictors_.size();
    outData += c * c0_;
    for (size_t k = 0; k < n; ++k)
        basePredictors_[k]->predictImpl_(inData, c * c1_[k], outData);

    size_t sampleCount = static_cast<size_t>(inData.rows());
    PROFILE::POP(sampleCount * n);
}

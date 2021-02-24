#include "pch.h"
#include "BoostPredictor.h"
#include "SimplePredictor.h"


BoostPredictor::~BoostPredictor() = default;


BoostPredictor::BoostPredictor(
    size_t variableCount,
    double c0,
    vector<double>&& c1,
    vector<unique_ptr<SimplePredictor>>&& basePredictors
) :
    variableCount_(variableCount),
    c0_{ c0 },
    c1_{ std::move(c1) },
    basePredictors_{ std::move(basePredictors) }
{
    ASSERT(c1_.size() == basePredictors_.size());
}


ArrayXd BoostPredictor::predict(CRefXXf inData) const
{
    //validateInData_(inData);
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred(sampleCount);
    pred = c0_;
    size_t n = basePredictors_.size();
    for (size_t k = 0; k < n; ++k)
        basePredictors_[k]->predict(inData, c1_[k], pred);
    return pred;
}

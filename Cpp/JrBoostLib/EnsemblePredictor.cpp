//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "EnsemblePredictor.h"


EnsemblePredictor::EnsemblePredictor(const vector<shared_ptr<Predictor>>& predictors) :
    predictors_(predictors)
{
    ASSERT(!predictors_.empty());
    for (const auto& pred : predictors_)
        ASSERT(pred->variableCount() == variableCount());
}

ArrayXd EnsemblePredictor::predict_(CRefXXf inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Zero(sampleCount);
    size_t n = size(predictors_);
    for (size_t k = 0; k < n; ++k)
        pred += predictors_[k]->predict_(inData);
    pred /= static_cast<double>(n);
    return pred;
}

void EnsemblePredictor::variableWeights_(vector<double>& weights, double c) const
{
    size_t n = size(predictors_);
    for (size_t k = 0; k < n; ++k)
        predictors_[k]->variableWeights_(weights, c / n);
}

void EnsemblePredictor::save_(ostream& os) const
{
    const int type = Ensemble;
    os.put(static_cast<char>(type));

    const uint32_t n = static_cast<uint32_t>(size(predictors_));
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (uint32_t i = 0; i < n; ++i)
        predictors_[i]->save_(os);
}

shared_ptr<Predictor> EnsemblePredictor::load_(istream& is, int version)
{
    if (version < 2) is.get();

    uint32_t n;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));

    vector<shared_ptr<Predictor>> predictors(n);
    for (uint32_t i = 0; i < n; ++i)
        predictors[i] = Predictor::load_(is, version);

    return std::make_shared<EnsemblePredictor>(predictors);
}

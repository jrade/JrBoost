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
    size_t n = predictors_.size();
    for (size_t k = 0; k < n; ++k)
        pred += predictors_[k]->predict_(inData);
    pred /= static_cast<double>(n);
    return pred;
}


void EnsemblePredictor::save_(ostream& os) const
{
    const uint8_t type = Ensemble;
    os.put(static_cast<char>(type));

    const uint8_t version = 1;
    os.put(static_cast<char>(version));

    const uint32_t n = static_cast<uint32_t>(predictors_.size());
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (uint32_t i = 0; i < n; ++i)
        predictors_[i]->save_(os);
}

shared_ptr<Predictor> EnsemblePredictor::load_(istream& is)
{
    uint8_t version = static_cast<uint8_t>(is.get());
    if (version != 1)
        parseError_();

    uint32_t n;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    vector<shared_ptr<Predictor>> predictors(n);
    for (uint32_t i = 0; i < n; ++i)
        predictors[i] = Predictor::load_(is);
    return std::make_shared<EnsemblePredictor>(predictors);
}

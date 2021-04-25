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

ArrayXd EnsemblePredictor::predictImpl_(CRefXXf inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Zero(sampleCount);
    size_t n = predictors_.size();
    for (size_t k = 0; k < n; ++k)
        pred += predictors_[k]->predictImpl_(inData);
    pred /= static_cast<double>(n);
    return pred;
}


void EnsemblePredictor::save(ostream& os) const
{
    const uint8_t type = Ensemble;
    os.write(reinterpret_cast<const char*>(&type), sizeof(type));

    const uint8_t version = 1;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));

    uint32_t n = static_cast<uint32_t>(predictors_.size());
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (uint32_t i = 0; i < n; ++i)
        predictors_[i]->save(os);
}

shared_ptr<Predictor> EnsemblePredictor::loadImpl_(istream& is)
{
    uint8_t version;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1)
        throw runtime_error("Not a valid JrBoost predictor file.");

    uint32_t n;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    vector<shared_ptr<Predictor>> predictors(n);
    for (uint32_t i = 0; i < n; ++i)
        predictors[i] = load(is);
    return std::make_shared<EnsemblePredictor>(predictors);
}

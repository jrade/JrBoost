//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostPredictor.h"
#include "BasePredictor.h"


BoostPredictor::~BoostPredictor() = default;

BoostPredictor::BoostPredictor(
    size_t variableCount,
    double c0,
    double c1,
    vector<unique_ptr<BasePredictor>>&& basePredictors
) :
    variableCount_(variableCount),
    c0_{ static_cast<float>(c0) },
    c1_{ static_cast<float>(c1) },
    basePredictors_{ move(basePredictors) }
{
}


ArrayXd BoostPredictor::predict_(CRefXXf inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Constant(sampleCount, static_cast<double>(c0_));
    size_t n = size(basePredictors_);
    for (size_t k = 0; k < n; ++k)
        basePredictors_[k]->predict_(inData, static_cast<double>(c1_), pred);
    pred = 1.0 / (1.0 + (-pred).exp());
    return pred;
}

void BoostPredictor::variableWeights_(vector<double>& weights, double c) const
{
    size_t n = size(basePredictors_);
    for (size_t k = 0; k < n; ++k)
        basePredictors_[k]->variableWeights_(weights, c * c1_);
}

void BoostPredictor::save_(ostream& os) const
{
    const int type = Boost;
    os.put(static_cast<char>(type));

    const uint32_t variableCount = static_cast<uint32_t>(variableCount_);
    os.write(reinterpret_cast<const char*>(&variableCount), sizeof(variableCount));
    os.write(reinterpret_cast<const char*>(&c0_), sizeof(c0_));
    os.write(reinterpret_cast<const char*>(&c1_), sizeof(c1_));

    const uint32_t n = static_cast<uint32_t>(size(basePredictors_));
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (uint32_t i = 0; i < n; ++i)
        basePredictors_[i]->save_(os);
}

shared_ptr<Predictor> BoostPredictor::load_(istream& is, int version)
{
    if (version < 2) is.get();

    uint32_t variableCount;
    float c0;
    float c1;
    is.read(reinterpret_cast<char*>(&variableCount), sizeof(variableCount));
    is.read(reinterpret_cast<char*>(&c0), sizeof(c0));
    is.read(reinterpret_cast<char*>(&c1), sizeof(c1));

    uint32_t n;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    vector<unique_ptr<BasePredictor>> basePredictors(n);
    for (uint32_t i = 0; i < n; ++i)
        basePredictors[i] = BasePredictor::load_(is, version);

    return std::make_shared<BoostPredictor>(variableCount, c0, c1, move(basePredictors));
}

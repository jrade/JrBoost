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
    vector<double>&& c1,
    vector<unique_ptr<BasePredictor>>&& basePredictors
) :
    variableCount_(variableCount),
    c0_{ c0 },
    c1_{ std::move(c1) },
    basePredictors_{ std::move(basePredictors) }
{
    ASSERT(c1_.size() == basePredictors_.size());
}


ArrayXd BoostPredictor::predictImpl_(CRefXXf inData) const
{
    PROFILE::PUSH(PROFILE::BOOST_PREDICT);

    validateInData(inData);
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Constant(sampleCount, c0_);
    size_t n = basePredictors_.size();
    for (size_t k = 0; k < n; ++k)
        basePredictors_[k]->predictImpl_(inData, c1_[k], pred);
    pred = 1.0 / (1.0 + (-pred).exp());
    PROFILE::POP(sampleCount * n);
    return pred;
}


void BoostPredictor::save(ostream& os) const
{
    const uint8_t type = Boost;
    os.write(reinterpret_cast<const char*>(&type), sizeof(type));

    const uint8_t version = 1;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));

    uint32_t variableCount = static_cast<uint32_t>(variableCount_);
    uint32_t n = static_cast<uint32_t>(basePredictors_.size());

    os.write(reinterpret_cast<const char*>(&variableCount), sizeof(variableCount));
    os.write(reinterpret_cast<const char*>(&c0_), sizeof(c0_));
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (uint32_t i = 0; i < n; ++i) {
        os.write(reinterpret_cast<const char*>(&c1_[i]), sizeof(c1_[i]));
        basePredictors_[i]->save(os);
    }
}

shared_ptr<Predictor> BoostPredictor::loadImpl_(istream& is)
{
    uint8_t version;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1)
        throw runtime_error("Not a valid JrBoost predictor file.");

    uint32_t variableCount;
    double c0;
    uint32_t n;
    vector<double> c1;
    vector<unique_ptr<BasePredictor>> basePredictors;

    is.read(reinterpret_cast<char*>(&variableCount), sizeof(variableCount));
    is.read(reinterpret_cast<char*>(&c0), sizeof(c0));
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    c1.resize(n);
    basePredictors.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        is.read(reinterpret_cast<char*>(&c1[i]), sizeof(c1[i]));
        basePredictors[i] = BasePredictor::load(is);
    }

    return shared_ptr<BoostPredictor>(new BoostPredictor(variableCount, c0, std::move(c1), std::move(basePredictors)));
}

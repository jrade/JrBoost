//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "Predictor.h"

#include "Base128Encoding.h"
#include "BasePredictor.h"


ArrayXd Predictor::predict(CRefXXfc inData) const
{
    PROFILE::PUSH(PROFILE::BOOST_PREDICT);

    if (static_cast<size_t>(inData.cols()) < variableCount())
        throw std::invalid_argument("Test indata has fewer variables than train indata.");
    if (!inData.isFinite().all())
        throw std::invalid_argument("Test indata has values that are infinity or NaN.");

    ArrayXd pred = predictImpl_(inData);

    PROFILE::POP();

    return pred;
}

ArrayXd Predictor::variableWeights() const
{
    ArrayXd weights = ArrayXd::Zero(variableCount());
    variableWeightsImpl_(1.0, weights);
    return weights;
}

shared_ptr<Predictor> Predictor::reindexVariables(const vector<size_t>& newIndices) const
{
    const size_t variableCount = *std::max_element(begin(newIndices), end(newIndices)) + 1;
    return reindexVariablesImpl_(newIndices, variableCount);
}


void Predictor::save(const string& filePath) const
{
    ofstream ofs;
    ofs.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    ofs.open(filePath, std::ios::binary);
    save(ofs);
}

void Predictor::save(ostream& os) const
{
    ASSERT(os.exceptions() == (std::ios::failbit | std::ios::badbit | std::ios::eofbit));

    // save header
    os.write("JRBOOST", 7);
    os.put(static_cast<char>(currentFileFormatVersion_));

    saveBody_(os);
    os.put('!');
}


shared_ptr<Predictor> Predictor::load(const string& filePath)
{
    ifstream ifs;
    ifs.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    ifs.open(filePath, std::ios::binary);
        return load(ifs);
}

shared_ptr<Predictor> Predictor::load(istream& is)
{
    ASSERT(is.exceptions() == (std::ios::failbit | std::ios::badbit | std::ios::eofbit));

    int version = loadHeader_(is);

    shared_ptr<Predictor> pred;
    try {
        pred = loadBody_(is, version);
    }
    catch (const std::overflow_error&) {
        parseError(is);
    }

    if (is.get() != '!')
        parseError(is);

    return pred;
}

int Predictor::loadHeader_(istream& is)
{
    char sig[7];
    is.read(sig, 7);
    if (memcmp(sig, "JRBOOST", 7) != 0)
        throw std::runtime_error("Not a JrBoost predictor file.");

    int version = is.get();
    if (version < 1)
        parseError(is);
    if (version > currentFileFormatVersion_)
        throw std::runtime_error("Reading this JrBoost predictor file requires a newer version of the JrBoost library.");

    return version;
}

shared_ptr<Predictor> Predictor::loadBody_(istream& is, int version)
{
    int type = is.get();
    if (version < 6) {
        if (type == 0)
            return BoostPredictor::loadBody_(is, version);
        if (type == 1)
            return EnsemblePredictor::loadBody_(is, version);
    }
    else {
        if (type == 'B')
            return BoostPredictor::loadBody_(is, version);
        if (type == 'E')
            return EnsemblePredictor::loadBody_(is, version);
        if (version >= 7 && type == 'U')
            return UnionPredictor::loadBody_(is, version);
    }
    parseError(is);
}


size_t Predictor::maxVariableCount_(const vector<shared_ptr<Predictor>>& predictors)
{
    size_t n = 0;
    for (const auto& predictor : predictors)
        n = std::max(n, predictor->variableCount());
    return n;
}

//----------------------------------------------------------------------------------------------------------------------

BoostPredictor::BoostPredictor(
    size_t variableCount,
    double c0,
    double c1,
    vector<unique_ptr<BasePredictor>>&& basePredictors
) :
    Predictor(variableCount),
    c0_{ static_cast<float>(c0) },
    c1_{ static_cast<float>(c1) },
    basePredictors_{ move(basePredictors) }
{
}

BoostPredictor::~BoostPredictor() = default;

shared_ptr<BoostPredictor> BoostPredictor::createInstance(
    size_t variableCount,
    double c0,
    double c1,
    vector<unique_ptr<BasePredictor>>&& basePredictors
)
{
    return makeShared<BoostPredictor>(variableCount, c0, c1, move(basePredictors));
}

ArrayXd BoostPredictor::predictImpl_(CRefXXfc inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Constant(sampleCount, static_cast<double>(c0_));
    for (const auto& basePredictor : basePredictors_)
        basePredictor->predict(inData, static_cast<double>(c1_), pred);
    return 1.0 / (1.0 + (-pred).exp());
}

void BoostPredictor::variableWeightsImpl_(double c, RefXd weights) const
{
    for (const auto& basePredictor : basePredictors_)
        basePredictor->variableWeights(c * c1_, weights);
}

shared_ptr<Predictor> BoostPredictor::reindexVariablesImpl_(const vector<size_t>& newIndices, size_t variableCount) const
{
    vector<unique_ptr<BasePredictor>> basePredictors;
    basePredictors.reserve(size(basePredictors_));
    for (const auto& basePredictor : basePredictors_)
        basePredictors.push_back(basePredictor->reindexVariables(newIndices));
    return createInstance(variableCount, c0_, c1_, move(basePredictors));
}

void BoostPredictor::saveBody_(ostream& os) const
{
    os.put('B');
    base128Save(os, variableCount());
    os.write(reinterpret_cast<const char*>(&c0_), sizeof(c0_));
    os.write(reinterpret_cast<const char*>(&c1_), sizeof(c1_));
    base128Save(os, size(basePredictors_));
    for (const auto& basePredictor : basePredictors_)
        basePredictor->save(os);
}

shared_ptr<BoostPredictor> BoostPredictor::loadBody_(istream& is, int version)
{
    if (version < 2) is.get();

    size_t variableCount;
    if (version < 5) {
        uint32_t vc32;
        is.read(reinterpret_cast<char*>(&vc32), sizeof(vc32));
        variableCount = static_cast<uint64_t>(vc32);
    }
    else
        variableCount = base128Load(is);

    float c0;
    float c1;
    is.read(reinterpret_cast<char*>(&c0), sizeof(c0));
    is.read(reinterpret_cast<char*>(&c1), sizeof(c1));

    size_t n;
    if (version < 5) {
        uint32_t n32;
        is.read(reinterpret_cast<char*>(&n32), sizeof(n32));
        n = static_cast<uint64_t>(n32);
    }
    else
        n = base128Load(is);

    vector<unique_ptr<BasePredictor>> basePredictors;
    basePredictors.reserve(n);
    for (; n != 0; --n)
        basePredictors.push_back(BasePredictor::load(is, version));

    return createInstance(variableCount, c0, c1, move(basePredictors));
}

//----------------------------------------------------------------------------------------------------------------------

EnsemblePredictor::EnsemblePredictor(const vector<shared_ptr<Predictor>>& predictors) :
    Predictor((
        ASSERT(!predictors.empty()),
        maxVariableCount_(predictors)
    )),
    predictors_(predictors)
{
}

shared_ptr<EnsemblePredictor> EnsemblePredictor::createInstance(const vector<shared_ptr<Predictor>>& predictors)
{
    return makeShared<EnsemblePredictor>(predictors);
}

ArrayXd EnsemblePredictor::predictImpl_(CRefXXfc inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Zero(sampleCount);
    for (const auto& predictor : predictors_)
        pred += predictor->predictImpl_(inData);
    pred /= static_cast<double>(size(predictors_));
    return pred;
}

void EnsemblePredictor::variableWeightsImpl_(double c, RefXd weights) const
{
    for (const auto& predictor : predictors_)
        predictor->variableWeightsImpl_(c / size(predictors_), weights);
}

shared_ptr<Predictor> EnsemblePredictor::reindexVariablesImpl_(const vector<size_t>& newIndices, size_t variableCount) const
{
    vector<shared_ptr<Predictor>> predictors;
    predictors.reserve(size(predictors_));
    for (const auto& predictor : predictors_)
        predictors.push_back(predictor->reindexVariablesImpl_(newIndices, variableCount));
    return createInstance(move(predictors));
}

void EnsemblePredictor::saveBody_(ostream& os) const
{
    os.put('E');
    base128Save(os, size(predictors_));
    for (const auto& predictor : predictors_)
        predictor->saveBody_(os);
}

shared_ptr<EnsemblePredictor> EnsemblePredictor::loadBody_(istream& is, int version)
{
    if (version < 2) is.get();

    size_t n;
    if (version < 5) {
        uint32_t n32;
        is.read(reinterpret_cast<char*>(&n32), sizeof(n32));
        n = static_cast<uint64_t>(n32);
    }
    else
        n = base128Load(is);

    vector<shared_ptr<Predictor>> predictors;
    predictors.reserve(n);
    for (; n != 0; --n)
        predictors.push_back(Predictor::loadBody_(is, version));

    return createInstance(predictors);
}

//----------------------------------------------------------------------------------------------------------------------

UnionPredictor::UnionPredictor(const vector<shared_ptr<Predictor>>& predictors) :
    Predictor(maxVariableCount_(predictors)),
    predictors_(predictors)
{
}

shared_ptr<UnionPredictor> UnionPredictor::createInstance(const vector<shared_ptr<Predictor>>& predictors)
{
    return makeShared<UnionPredictor>(predictors);
}

ArrayXd UnionPredictor::predictImpl_(CRefXXfc inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Zero(sampleCount);
    for (const auto& predictor : predictors_)
        pred += (1.0 - pred) * predictor->predictImpl_(inData);
    return pred;
}

void UnionPredictor::variableWeightsImpl_(double c, RefXd weights) const
{
    for (const auto& predictor : predictors_)
        predictor->variableWeightsImpl_(c, weights);
}

shared_ptr<Predictor> UnionPredictor::reindexVariablesImpl_(const vector<size_t>& newIndices, size_t variableCount) const
{
    vector<shared_ptr<Predictor>> predictors;
    predictors.reserve(size(predictors_));
    for (const auto& predictor : predictors_)
        predictors.push_back(predictor->reindexVariablesImpl_(newIndices, variableCount));
    return createInstance(predictors);
}

void UnionPredictor::saveBody_(ostream& os) const
{
    os.put('U');
    base128Save(os, size(predictors_));
    for (const auto& predictor : predictors_)
        predictor->saveBody_(os);
}

shared_ptr<UnionPredictor> UnionPredictor::loadBody_(istream& is, int version)
{
    size_t n = base128Load(is);
    vector<shared_ptr<Predictor>> predictors;
    predictors.reserve(n);
    for (; n != 0; --n)
        predictors.push_back(Predictor::loadBody_(is, version));
    return createInstance(predictors);
}

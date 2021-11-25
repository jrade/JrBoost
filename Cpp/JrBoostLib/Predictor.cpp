//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "Predictor.h"

#include "Base128Encoding.h"
#include "BasePredictor.h"


Predictor::Predictor(size_t variableCount) :
    variableCount_(variableCount)
{}

Predictor::~Predictor() = default;


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

ArrayXf Predictor::variableWeights() const
{
    return variableWeightsImpl_();
}

shared_ptr<const Predictor> Predictor::reindexVariables(const vector<size_t>& newIndices) const
{
    if (size(newIndices) < variableCount())
        throw std::runtime_error("The size of the new indices array"
            " must be greater than or equal to the variable count.");
    return reindexVariablesImpl_(newIndices);
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
    os.write("JRBOOST", 7);
    os.put(static_cast<char>(currentFileFormatVersion_));
    saveImpl_(os);
    os.put('!');
}

shared_ptr<const Predictor> Predictor::load(const string& filePath)
{
    ifstream ifs;
    ifs.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    ifs.open(filePath, std::ios::binary);
        return load(ifs);
}

shared_ptr<const Predictor> Predictor::load(istream& is)
{
    ASSERT(is.exceptions() == (std::ios::failbit | std::ios::badbit | std::ios::eofbit));

    try {
        char sig[7];
        is.read(sig, 7);
        if (memcmp(sig, "JRBOOST", 7) != 0)
            throw std::runtime_error("Not a JrBoost predictor file.");

        int version = is.get();
        if (version < 1)
            parseError(is);
        if (version > currentFileFormatVersion_)
            throw std::runtime_error("Reading this JrBoost predictor file requires a newer version of the JrBoost library.");

        shared_ptr<const Predictor> pred = loadImpl_(is, version);

        if (is.get() != '!')
            parseError(is);

        return pred;
    }
    catch (const std::ios::failure&) {
        parseError(is);
    }
    catch (const std::overflow_error&) {    // thrown by base128Load()
        parseError(is);
    }
}

shared_ptr<const Predictor> Predictor::loadImpl_(istream& is, int version)
{
    int type = is.get();
    if (version < 6) {
        if (type == 0)
            return BoostPredictor::loadImpl_(is, version);
        if (type == 1)
            return EnsemblePredictor::loadImpl_(is, version);
    }
    else {
        if (type == 'B')
            return BoostPredictor::loadImpl_(is, version);
        if (type == 'E')
            return EnsemblePredictor::loadImpl_(is, version);
        if (version >= 7 && type == 'U')
            return UnionPredictor::loadImpl_(is, version);
    }
    parseError(is);
}

//----------------------------------------------------------------------------------------------------------------------

shared_ptr<const Predictor> BoostPredictor::createInstance(
    double c0,
    double c1,
    vector<unique_ptr<const BasePredictor>>&& basePredictors
)
{
    return makeShared<BoostPredictor>(c0, c1, move(basePredictors));
}

BoostPredictor::BoostPredictor(
    double c0,
    double c1,
    vector<unique_ptr<const BasePredictor>>&& basePredictors
) :
    Predictor(initVariableCount_(basePredictors)),
    c0_{ static_cast<float>(c0) },
    c1_{ static_cast<float>(c1) },
    basePredictors_{ move(basePredictors) }
{
}

size_t BoostPredictor::initVariableCount_(const vector<unique_ptr<const BasePredictor>>& basePredictors)
{
    size_t n = 0;
    for (const auto& basePredictor : basePredictors)
        n = std::max(n, basePredictor->variableCount());
    return n;
}

BoostPredictor::~BoostPredictor() = default;


ArrayXd BoostPredictor::predictImpl_(CRefXXfc inData) const
{
    const size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Constant(sampleCount, static_cast<double>(c0_));
    for (const auto& basePredictor : basePredictors_)
        basePredictor->predict(inData, static_cast<double>(c1_), pred);
    return 1.0 / (1.0 + (-pred).exp());
}

ArrayXf BoostPredictor::variableWeightsImpl_() const
{
    ArrayXd weights = ArrayXd::Zero(variableCount());
    const double c = 1.0 / size(basePredictors_);
    for (const auto& basePredictor : basePredictors_)
        basePredictor->variableWeights(c, weights);
    return weights.cast<float>();
}

shared_ptr<const Predictor> BoostPredictor::reindexVariablesImpl_(const vector<size_t>& newIndices) const
{
    vector<unique_ptr<const BasePredictor>> basePredictors;
    basePredictors.reserve(size(basePredictors_));
    for (const auto& basePredictor : basePredictors_)
        basePredictors.push_back(basePredictor->reindexVariables(newIndices));
    return createInstance(c0_, c1_, move(basePredictors));
}

void BoostPredictor::saveImpl_(ostream& os) const
{
    os.put('B');
    os.write(reinterpret_cast<const char*>(&c0_), sizeof(c0_));
    os.write(reinterpret_cast<const char*>(&c1_), sizeof(c1_));
    base128Save(os, size(basePredictors_));
    for (const auto& basePredictor : basePredictors_)
        basePredictor->save(os);
}

shared_ptr<const Predictor> BoostPredictor::loadImpl_(istream& is, int version)
{
    if (version < 2)
        is.ignore(5);
    else if (version < 5)
        is.ignore(4);
    else if (version < 8)
        base128Load(is);

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

    vector<unique_ptr<const BasePredictor>> basePredictors;
    basePredictors.reserve(n);
    for (; n != 0; --n)
        basePredictors.push_back(BasePredictor::load(is, version));

    return createInstance(c0, c1, move(basePredictors));
}

//----------------------------------------------------------------------------------------------------------------------

shared_ptr<const Predictor> EnsemblePredictor::createInstance(const vector<shared_ptr<const Predictor>>& predictors)
{
    return makeShared<EnsemblePredictor>(predictors);
}

EnsemblePredictor::EnsemblePredictor(const vector<shared_ptr<const Predictor>>& predictors
) :
    Predictor(initVariableCount_(predictors)),
    predictors_(predictors)
{
}

size_t EnsemblePredictor::initVariableCount_(const vector<shared_ptr<const Predictor>>& predictors)
{
    ASSERT(!predictors.empty());
    size_t n = 0;
    for (const auto& predictor : predictors)
        n = std::max(n, predictor->variableCount());
    return n;
}

EnsemblePredictor::~EnsemblePredictor() = default;


ArrayXd EnsemblePredictor::predictImpl_(CRefXXfc inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Zero(sampleCount);
    for (const auto& predictor : predictors_)
        pred += predictor->predictImpl_(inData);
    pred /= static_cast<double>(size(predictors_));
    return pred;
}

ArrayXf EnsemblePredictor::variableWeightsImpl_() const
{
    ArrayXf weights = ArrayXf::Zero(variableCount());
    for (const auto& predictor : predictors_)
        weights += predictor->variableWeightsImpl_();
    weights /= static_cast<float>(size(predictors_));
    return weights;
}

shared_ptr<const Predictor> EnsemblePredictor::reindexVariablesImpl_(const vector<size_t>& newIndices) const
{
    vector<shared_ptr<const Predictor>> predictors;
    predictors.reserve(size(predictors_));
    for (const auto& predictor : predictors_)
        predictors.push_back(predictor->reindexVariablesImpl_(newIndices));
    return createInstance(move(predictors));
}

void EnsemblePredictor::saveImpl_(ostream& os) const
{
    os.put('E');
    base128Save(os, size(predictors_));
    for (const auto& predictor : predictors_)
        predictor->saveImpl_(os);
}

shared_ptr<const Predictor> EnsemblePredictor::loadImpl_(istream& is, int version)
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

    vector<shared_ptr<const Predictor>> predictors;
    predictors.reserve(n);
    for (; n != 0; --n)
        predictors.push_back(Predictor::loadImpl_(is, version));

    return createInstance(predictors);
}

//----------------------------------------------------------------------------------------------------------------------

shared_ptr<const Predictor> UnionPredictor::createInstance(const vector<shared_ptr<const Predictor>>& predictors)
{
    return makeShared<UnionPredictor>(predictors);
}

UnionPredictor::UnionPredictor(const vector<shared_ptr<const Predictor>>& predictors) :
    Predictor(initVariableCount_(predictors)),
    predictors_(predictors)
{
}

size_t UnionPredictor::initVariableCount_(const vector<shared_ptr<const Predictor>>& predictors)
{
    size_t n = 0;
    for (const auto& predictor : predictors)
        n = std::max(n, predictor->variableCount());
    return n;
}

UnionPredictor::~UnionPredictor() = default;


ArrayXd UnionPredictor::predictImpl_(CRefXXfc inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Zero(sampleCount);
    for (const auto& predictor : predictors_)
        pred += (1.0 - pred) * predictor->predictImpl_(inData);
    return pred;
}

ArrayXf UnionPredictor::variableWeightsImpl_() const
{
    ArrayXf weights = ArrayXf::Zero(variableCount());
    for (const auto& predictor : predictors_)
        weights += predictor->variableWeightsImpl_();
    return weights;
}

shared_ptr<const Predictor> UnionPredictor::reindexVariablesImpl_(const vector<size_t>& newIndices) const
{
    vector<shared_ptr<const Predictor>> predictors;
    predictors.reserve(size(predictors_));
    for (const auto& predictor : predictors_)
        predictors.push_back(predictor->reindexVariablesImpl_(newIndices));
    return createInstance(predictors);
}

void UnionPredictor::saveImpl_(ostream& os) const
{
    os.put('U');
    base128Save(os, size(predictors_));
    for (const auto& predictor : predictors_)
        predictor->saveImpl_(os);
}

shared_ptr<const Predictor> UnionPredictor::loadImpl_(istream& is, int version)
{
    size_t n = base128Load(is);
    vector<shared_ptr<const Predictor>> predictors;
    predictors.reserve(n);
    for (; n != 0; --n)
        predictors.push_back(Predictor::loadImpl_(is, version));
    return createInstance(predictors);
}

//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "Predictor.h"

#include "Base128Encoding.h"
#include "BasePredictor.h"
#include "OmpParallel.h"


Predictor::Predictor(size_t variableCount) : variableCount_(variableCount) {}


ArrayXd Predictor::predict(CRefXXfc inData, size_t threadCount) const
{
    PROFILE::PUSH(PROFILE::PREDICT);

    if (::currentInterruptHandler != nullptr)
        ::currentInterruptHandler->check();
    if (abortThreads)
        throw ThreadAborted();

    if (static_cast<size_t>(inData.cols()) < variableCount())
        throw std::invalid_argument("Test indata has fewer variables than train indata.");
    if (!inData.isFinite().all())
        throw std::invalid_argument("Test indata has values that are infinity or NaN.");

    if (threadCount == 0 || threadCount > omp_get_max_threads())
        threadCount = omp_get_max_threads();

    ArrayXd pred = predictImpl_(inData, threadCount);

    PROFILE::SWITCH(PROFILE::ZERO);   // calibrate the profiling
    PROFILE::POP();
    return pred;
}

double Predictor::predictOne(CRefXf inData) const
{
    if (static_cast<size_t>(inData.rows()) < variableCount())
        throw std::invalid_argument("Test indata has fewer variables than train indata.");
    if (!inData.isFinite().all())
        throw std::invalid_argument("Test indata has values that are infinity or NaN.");

    return predictOneImpl_(inData);
}

ArrayXf Predictor::variableWeights() const { return variableWeightsImpl_(); }

shared_ptr<Predictor> Predictor::reindexVariables(CRefXs newIndices) const
{
    if (static_cast<size_t>(newIndices.rows()) < variableCount())
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

    try {
        char sig[7];
        is.read(sig, 7);
        if (memcmp(sig, "JRBOOST", 7) != 0)
            throw std::runtime_error("Not a JrBoost predictor file.");

        int version = is.get();
        if (version < 1)
            parseError(is);
        if (version > currentFileFormatVersion_)
            throw std::runtime_error(
                "Reading this JrBoost predictor file requires a newer version of the JrBoost library.");

        shared_ptr<Predictor> pred = loadImpl_(is, version);

        if (is.get() != '!')
            parseError(is);

        return pred;
    }
    catch (const std::ios::failure&) {
        parseError(is);
    }
    catch (const std::overflow_error&) {   // thrown by base128Load()
        parseError(is);
    }
}

shared_ptr<Predictor> Predictor::loadImpl_(istream& is, int version)
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

shared_ptr<Predictor>
BoostPredictor::createInstance(double c0, double c1, vector<unique_ptr<BasePredictor>>&& basePredictors)
{
    return makeShared<BoostPredictor>(c0, c1, move(basePredictors));
}

BoostPredictor::BoostPredictor(double c0, double c1, vector<unique_ptr<BasePredictor>>&& basePredictors) :
    Predictor(initVariableCount_(basePredictors)),
    c0_{static_cast<float>(c0)},
    c1_{static_cast<float>(c1)},
    basePredictors_{move(basePredictors)}
{
}

size_t BoostPredictor::initVariableCount_(const vector<unique_ptr<BasePredictor>>& basePredictors)
{
    size_t n = 0;
    for (const auto& basePredictor : basePredictors)
        n = std::max(n, basePredictor->variableCount_());
    return n;
}

BoostPredictor::~BoostPredictor() = default;


ArrayXd BoostPredictor::predictImpl_(CRefXXfc inData, size_t threadCount) const
{
    if (threadCount == 1)
        return predictImplNoThreads_(inData);

    const size_t sampleCount = static_cast<size_t>(inData.rows());
    const size_t basePredictorCount = size(basePredictors_);
    threadCount = std::min(threadCount, basePredictorCount);
    const size_t padding = std::hardware_destructive_interference_size / sizeof(double);

    static thread_local vector<double> buf;
    buf.assign((sampleCount + padding) * threadCount, 0.0);
    Eigen::Map<ArrayXXdc> paddedPredByThread(data(buf), sampleCount + padding, threadCount);
    RefXXdc predByThread(paddedPredByThread(Eigen::seqN(0, sampleCount), Eigen::all));
    std::atomic<size_t> nextK = 0;

    BEGIN_OMP_PARALLEL(threadCount)
    {
        const size_t id = omp_get_thread_num();
        while (true) {
            size_t k = nextK++;
            if (k >= basePredictorCount)
                break;
            basePredictors_[k]->predict_(inData, static_cast<double>(c1_), predByThread.col(id));
        }
    }
    END_OMP_PARALLEL

    ArrayXd pred = static_cast<double>(c0_) + predByThread.rowwise().sum();
    return (1.0 + (-pred).exp()).inverse();
}

ArrayXd BoostPredictor::predictImplNoThreads_(CRefXXfc inData) const
{
    const size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Constant(sampleCount, static_cast<double>(c0_));
    for (const auto& basePredictor : basePredictors_)
        basePredictor->predict_(inData, static_cast<double>(c1_), pred);
    return (1.0 + (-pred).exp()).inverse();
}

double BoostPredictor::predictOneImpl_(CRefXf inData) const
{
    double pred = c0_;
    for (const auto& basePredictor : basePredictors_)
        pred += c1_ * basePredictor->predictOne_(inData);
    return 1.0 / (1.0 + std::exp(-pred));
}

ArrayXf BoostPredictor::variableWeightsImpl_() const
{
    ArrayXd weights = ArrayXd::Zero(variableCount());
    const double c = 1.0 / size(basePredictors_);
    for (const auto& basePredictor : basePredictors_)
        basePredictor->variableWeights_(c, weights);
    return weights.cast<float>();
}

shared_ptr<Predictor> BoostPredictor::reindexVariablesImpl_(CRefXs newIndices) const
{
    vector<unique_ptr<BasePredictor>> basePredictors;
    basePredictors.reserve(size(basePredictors_));
    for (const auto& basePredictor : basePredictors_)
        basePredictors.push_back(basePredictor->reindexVariables_(newIndices));
    return createInstance(c0_, c1_, move(basePredictors));
}


void BoostPredictor::saveImpl_(ostream& os) const
{
    os.put('B');
    os.write(reinterpret_cast<const char*>(&c0_), sizeof(c0_));
    os.write(reinterpret_cast<const char*>(&c1_), sizeof(c1_));
    base128Save(os, size(basePredictors_));
    for (const auto& basePredictor : basePredictors_)
        basePredictor->save_(os);
}

shared_ptr<Predictor> BoostPredictor::loadImpl_(istream& is, int version)
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

    vector<unique_ptr<BasePredictor>> basePredictors;
    basePredictors.reserve(n);
    for (; n != 0; --n)
        basePredictors.push_back(BasePredictor::load_(is, version));

    return createInstance(c0, c1, move(basePredictors));
}

//----------------------------------------------------------------------------------------------------------------------

shared_ptr<Predictor> EnsemblePredictor::createInstance(const vector<shared_ptr<Predictor>>& predictors)
{
    return makeShared<EnsemblePredictor>(predictors);
}

EnsemblePredictor::EnsemblePredictor(const vector<shared_ptr<Predictor>>& predictors) :
    Predictor(initVariableCount_(predictors)), predictors_(predictors)
{
}

size_t EnsemblePredictor::initVariableCount_(const vector<shared_ptr<Predictor>>& predictors)
{
    ASSERT(!predictors.empty());
    size_t n = 0;
    for (const auto& predictor : predictors)
        n = std::max(n, predictor->variableCount());
    return n;
}


ArrayXd EnsemblePredictor::predictImpl_(CRefXXfc inData, size_t threadCount) const
{
    if (threadCount == 1)
        return predictImplNoThreads_(inData);

    const size_t sampleCount = static_cast<size_t>(inData.rows());
    const size_t predictorCount = size(predictors_);
    const size_t outerThreadCount = std::min(threadCount, predictorCount);
    const size_t padding = std::hardware_destructive_interference_size / sizeof(double);

    ArrayXXdc paddedPredByThread = ArrayXXdc::Zero(sampleCount + padding, outerThreadCount);
    RefXXdc predByThread(paddedPredByThread(Eigen::seqN(0, sampleCount), Eigen::all));
    std::atomic<size_t> nextK = 0;

    BEGIN_OMP_PARALLEL(outerThreadCount)
    {
        const size_t id = omp_get_thread_num();
        const size_t innerThreadCount
            = (threadCount * (id + 1)) / outerThreadCount - (threadCount * id) / outerThreadCount;
        while (true) {
            size_t k = nextK++;
            if (k >= predictorCount)
                break;
            predByThread.col(id) += predictors_[k]->predictImpl_(inData, innerThreadCount);
        }
    }
    END_OMP_PARALLEL

    return predByThread.rowwise().sum() / static_cast<double>(predictorCount);
}

ArrayXd EnsemblePredictor::predictImplNoThreads_(CRefXXfc inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Zero(sampleCount);
    for (const auto& predictor : predictors_)
        pred += predictor->predictImplNoThreads_(inData);
    pred /= static_cast<double>(size(predictors_));
    return pred;
}

double EnsemblePredictor::predictOneImpl_(CRefXf inData) const
{
    double pred = 0.0;
    for (const auto& predictor : predictors_)
        pred += predictor->predictOneImpl_(inData);
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

shared_ptr<Predictor> EnsemblePredictor::reindexVariablesImpl_(CRefXs newIndices) const
{
    vector<shared_ptr<Predictor>> predictors;
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

shared_ptr<Predictor> EnsemblePredictor::loadImpl_(istream& is, int version)
{
    if (version < 2)
        is.get();

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
        predictors.push_back(Predictor::loadImpl_(is, version));

    return createInstance(predictors);
}

//----------------------------------------------------------------------------------------------------------------------

shared_ptr<Predictor> UnionPredictor::createInstance(const vector<shared_ptr<Predictor>>& predictors)
{
    return makeShared<UnionPredictor>(predictors);
}

UnionPredictor::UnionPredictor(const vector<shared_ptr<Predictor>>& predictors) :
    Predictor(initVariableCount_(predictors)), predictors_(predictors)
{
}

size_t UnionPredictor::initVariableCount_(const vector<shared_ptr<Predictor>>& predictors)
{
    size_t n = 0;
    for (const auto& predictor : predictors)
        n = std::max(n, predictor->variableCount());
    return n;
}


ArrayXd UnionPredictor::predictImpl_(CRefXXfc inData, size_t threadCount) const
{
    if (threadCount == 1)
        return predictImplNoThreads_(inData);

    const size_t sampleCount = static_cast<size_t>(inData.rows());
    const size_t predictorCount = size(predictors_);
    const size_t outerThreadCount = std::min(threadCount, predictorCount);
    const size_t padding = std::hardware_destructive_interference_size / sizeof(double);

    ArrayXXdc paddedPredByThread = ArrayXXdc::Ones(sampleCount + padding, outerThreadCount);
    RefXXdc predByThread(paddedPredByThread(Eigen::seqN(0, sampleCount), Eigen::all));
    std::atomic<size_t> nextK = 0;

    BEGIN_OMP_PARALLEL(outerThreadCount)
    {
        const size_t id = omp_get_thread_num();
        const size_t innerThreadCount
            = (threadCount * (id + 1)) / outerThreadCount - (threadCount * id) / outerThreadCount;
        while (true) {
            size_t k = nextK++;
            if (k >= predictorCount)
                break;
            predByThread.col(id) *= 1.0 - predictors_[k]->predictImpl_(inData, innerThreadCount);
        }
    }
    END_OMP_PARALLEL

    return 1.0 - predByThread.rowwise().prod();
}

ArrayXd UnionPredictor::predictImplNoThreads_(CRefXXfc inData) const
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    ArrayXd pred = ArrayXd::Ones(sampleCount);
    for (const auto& predictor : predictors_)
        pred *= 1.0 - predictor->predictImplNoThreads_(inData);
    return 1.0 - pred;
}

double UnionPredictor::predictOneImpl_(CRefXf inData) const
{
    double pred = 1.0;
    for (const auto& predictor : predictors_)
        pred *= 1.0 - predictor->predictOneImpl_(inData);
    return 1.0 - pred;
}

ArrayXf UnionPredictor::variableWeightsImpl_() const
{
    ArrayXf weights = ArrayXf::Zero(variableCount());
    for (const auto& predictor : predictors_)
        weights += predictor->variableWeightsImpl_();
    return weights;
}

shared_ptr<Predictor> UnionPredictor::reindexVariablesImpl_(CRefXs newIndices) const
{
    vector<shared_ptr<Predictor>> predictors;
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

shared_ptr<Predictor> UnionPredictor::loadImpl_(istream& is, int version)
{
    size_t n = base128Load(is);
    vector<shared_ptr<Predictor>> predictors;
    predictors.reserve(n);
    for (; n != 0; --n)
        predictors.push_back(Predictor::loadImpl_(is, version));
    return createInstance(predictors);
}

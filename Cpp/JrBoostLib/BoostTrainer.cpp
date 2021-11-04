//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostTrainer.h"

#include "BoostOptions.h"
#include "BoostPredictor.h"
#include "FastExp.h"
#include "ForestTrainer.h"


BoostTrainer::BoostTrainer(ArrayXXfc inData, ArrayXu8 outData, optional<ArrayXd> weights) :
    sampleCount_{(
        validateData_(inData, outData, weights),        // do validation before everything else
        static_cast<size_t>(inData.rows())
    )},
    variableCount_{ static_cast<size_t>(inData.cols()) },
    inData_{ std::move(inData)},
    outData_{ 2.0 * outData.cast<double>() - 1.0 },
    weights_{ std::move(weights) },
    strata_{ std::move(outData) },
    globaLogOddsRatio_{ getGlobalLogOddsRatio_() },
    baseTrainer_{ std::make_unique<ForestTrainer>(inData_, strata_) }
{
}

BoostTrainer::~BoostTrainer() = default;


void BoostTrainer::validateData_(CRefXXfc inData, CRefXu8 outData, optional<CRefXd> weights)
{
    const size_t sampleCount = static_cast<size_t>(inData.rows());
    const size_t variableCount = static_cast<size_t>(inData.cols());

    if (sampleCount == 0)
        throw std::invalid_argument("Train indata has 0 samples.");
    if (variableCount == 0)
        throw std::invalid_argument("Train indata has 0 variables.");
    if (!(inData.abs() < numeric_limits<float>::infinity()).all())          // carefully written to trap NaN
        throw std::invalid_argument("Train indata has values that are infinity or NaN.");

    if (static_cast<size_t>(outData.rows()) != sampleCount)
        throw std::invalid_argument("Train indata and outdata have different numbers of samples.");
    if ((outData > 1).any())
        throw std::invalid_argument("Train outdata has values that are not 0 or 1.");

    if (weights) {
        if (static_cast<size_t>(weights->rows()) != sampleCount)
            throw std::invalid_argument("Train indata and weights have different numbers of samples.");
        if (!(weights->abs() < numeric_limits<double>::infinity()).all())   // carefully written to trap NaN
            throw std::invalid_argument("Train weights have values that are infinity or NaN.");
        if ((*weights <= 0.0).any())
            throw std::invalid_argument("Train weights have non-positive values.");
    }
}


double BoostTrainer::getGlobalLogOddsRatio_() const
{
    double p0, p1;

    if (!weights_) {
        p0 = (1.0 - outData_).sum();
        p1 = (1.0 + outData_).sum();
    }
    else {
        p0 = ((*weights_) * (1.0 - outData_)).sum();
        p1 = ((*weights_) * (1.0 + outData_)).sum();
    }

    if (p0 == 0.0)
        throw std::invalid_argument("There are no train samples with label 0.");
    if (p1 == 0.0)
        throw std::invalid_argument("There are no train samples with label 1.");

    return std::log(p1) - std::log(p0);
}

//----------------------------------------------------------------------------------------------------------------------

shared_ptr<BoostPredictor> BoostTrainer::train(const BoostOptions& opt, size_t threadCount) const
{
    PROFILE::PUSH(PROFILE::BOOST_TRAIN);
    const size_t ITEM_COUNT = sampleCount_ * opt.iterationCount();

    shared_ptr<BoostPredictor> pred;

    double gamma = opt.gamma();
    if (gamma == 1.0)
        pred = trainAda_(opt, threadCount);
    else if (gamma == 0.0)
        pred = trainLogit_(opt, threadCount);
    else
        pred = trainRegularizedLogit_(opt, threadCount);
    
    PROFILE::POP(ITEM_COUNT);

    return pred;
}

//......................................................................................................................

shared_ptr<BoostPredictor> BoostTrainer::trainAda_(const BoostOptions& opt, size_t threadCount) const
{
    const size_t sampleCount = sampleCount_;
    const size_t iterationCount = opt.iterationCount();
    const double eta = opt.eta();

    ArrayXd adjWeights(sampleCount);
    ArrayXd F = ArrayXd::Constant(sampleCount, globaLogOddsRatio_ / 2.0);

    const double* pOutData = std::data(outData_);
    const double* pWeights = weights_ ? std::data(*weights_) : nullptr;
    double* pAdjWeights = std::data(adjWeights);
    double* pF = std::data(F);

    vector<unique_ptr<BasePredictor>> basePredictors(iterationCount);

    for (size_t k = 0; k != iterationCount; ++k) {

        double adjWeightSum = 0.0;

        // Add all the adjusted weights to check if any of them is infinity.
        // Taking maximum might be more natural than adding,
        // but it seems Visual C++ can autovectorize sum reduction but not max reduction.

        if (!opt.fastExp()) {

            // Visual C++ does autovectorize the following two loops

            if (pWeights == nullptr) {
                for (size_t i = 0; i < sampleCount; ++i) {
                    double x = std::exp(-pF[i] * pOutData[i]);
                    pAdjWeights[i] = x;
                    adjWeightSum += x;
                }
            }
            else {
                for (size_t i = 0; i < sampleCount; ++i) {
                    double x = std::exp(-pF[i] * pOutData[i]);
                    x *= pWeights[i];
                    pAdjWeights[i] = x;
                    adjWeightSum += x;
                }
            }
        }

        else {

            // Visual C++ fails to autovectorize the fastExp() function with AVX2
            // (or the fastExp() function with AVX512F + AVX512DQ). Thus we do manual vectorization.

#if defined(__AVX512F__) && defined(__AVX512DQ__)

            // WARNING: NOT TESTED!

            size_t i = 0;
            __m512d adjWeightSum8 = _mm512_setzero_pd();

            if (pWeights == nullptr) {
                while (i + 8 <= sampleCount) {
                    const __m512d F8 = _mm512_load_pd(pF + i);
                    const __m512d y8 = _mm512_load_pd(pOutData + i);
                    __m512d x8 = _mm512_mul_pd(F8, y8);
                    x8 = _mm512_xor_pd(x8, _mm512_set1_pd(-0.0));       // x8 = -x8
                    x8 = fastExp(x8);
                    _mm512_store_pd(pAdjWeights + i, x8);
                    adjWeightSum8 = _mm512_add_pd(adjWeightSum8, x8);
                    i += 8;
                }
            }
            else {
                while (i + 8 <= sampleCount) {
                    const __m512d F8 = _mm512_load_pd(pF + i);
                    const __m512d y8 = _mm512_load_pd(pOutData + i);
                    const __m512d w8 = _mm512_loadu_pd(pWeights + i);   // allocated by client, alignment unknown
                    __m512d x8 = _mm512_mul_pd(F8, y8);
                    x8 = _mm512_xor_pd(x8, _mm512_set1_pd(-0.0));       // x8 = -x8
                    x8 = fastExp(x8);
                    x8 = _mm512_mul_pd(x8, w8);
                    _mm512_store_pd(pAdjWeights + i, x8);
                    adjWeightSum8 = _mm512_add_pd(adjWeightSum8, x8);
                    i += 8;
                }
            }

            adjWeightSum = std::accumulate(std::begin(adjWeightSum8.m512d_f64), std::end(adjWeightSum8.m512d_f64), 0.0);

            if (pWeights == nullptr) {
                while (i < sampleCount) {
                    double x = fastExp(-pF[i] * pOutData[i]);
                    pAdjWeights[i] = x;
                    adjWeightSum += x;
                    ++i;
                }
            }
            else {
                while (i < sampleCount) {
                    const double x = fastExp(-pF[i] * pOutData[i]);
                    x *= pWeights[i];
                    pAdjWeights[i] = x;
                    adjWeightSum += x;
                    ++i;
                }
            }

#elif defined(__AVX2__)

            size_t i = 0;
            __m256d adjWeightSum4 = _mm256_setzero_pd();

            if (pWeights == nullptr) {
                while (i + 4 <= sampleCount) {
                    const  __m256d F4 = _mm256_load_pd(pF + i);
                    const __m256d y4 = _mm256_load_pd(pOutData + i);
                    __m256d x4 = _mm256_mul_pd(F4, y4);
                    x4 = _mm256_xor_pd(x4, _mm256_set1_pd(-0.0));       // x4 = -x4
                    x4 = fastExp(x4);
                    _mm256_store_pd(pAdjWeights + i, x4);
                    adjWeightSum4 = _mm256_add_pd(adjWeightSum4, x4);
                    i += 4;
                }
            }
            else {
                while (i + 4 <= sampleCount) {
                    const __m256d F4 = _mm256_load_pd(pF + i);
                    const __m256d y4 = _mm256_load_pd(pOutData + i);
                    const __m256d w4 = _mm256_loadu_pd(pWeights + i);   // allocated by client, alignment unknown
                    __m256d x4 = _mm256_mul_pd(F4, y4);
                    x4 = _mm256_xor_pd(x4, _mm256_set1_pd(-0.0));       // x4 = -x4
                    x4 = fastExp(x4);
                    x4 = _mm256_mul_pd(x4, w4);
                    _mm256_store_pd(pAdjWeights + i, x4);
                    adjWeightSum4 = _mm256_add_pd(adjWeightSum4, x4);
                    i += 4;
                }
            }

            adjWeightSum = std::accumulate(std::begin(adjWeightSum4.m256d_f64), std::end(adjWeightSum4.m256d_f64), 0.0);

            if (pWeights == nullptr) {
                while (i < sampleCount) {
                    double x = fastExp(-pF[i] * pOutData[i]);
                    pAdjWeights[i] = x;
                    adjWeightSum += x;
                    ++i;
                }
            }
            else {
                while (i < sampleCount) {
                    double x = fastExp(-pF[i] * pOutData[i]);
                    x *= pWeights[i];
                    pAdjWeights[i] = x;
                    adjWeightSum += x;
                    ++i;
                }
            }

#else
            if (pWeights == nullptr) {
                for (size_t i = 0; i < sampleCount; ++i) {
                    double x = fastExp(-pF[i] * pOutData[i]);
                    pAdjWeights[i] = x;
                    adjWeightSum += x;
                }
            }
            else {
                for (size_t i = 0; i < sampleCount; ++i) {
                    double x = fastExp(-pF[i] * pOutData[i]);
                    x *= pWeights[i];
                    pAdjWeights[i] = x;
                    adjWeightSum += x;
                }
            }
#endif
        }  // end if (!opt.fastExp())

        if (!(adjWeightSum < numeric_limits<double>::infinity()))   // carefully written to trap NaN
            overflow_(opt);

        unique_ptr<BasePredictor> basePred = baseTrainer_->train(outData_, adjWeights, opt, threadCount);
        basePred->predict(inData_, eta, F);
        basePredictors[k] = move(basePred);
    }

    return std::make_shared<BoostPredictor>(variableCount_, globaLogOddsRatio_, 2 * eta, move(basePredictors));
}

//......................................................................................................................

shared_ptr<BoostPredictor> BoostTrainer::trainLogit_(const BoostOptions& opt, size_t threadCount) const
{
    const size_t sampleCount = sampleCount_;
    const size_t iterationCount = opt.iterationCount();
    const double eta = opt.eta();

    ArrayXd adjOutData(sampleCount);
    ArrayXd adjWeights(sampleCount);
    ArrayXd F = ArrayXd::Constant(sampleCount, globaLogOddsRatio_);

    const double* pOutData = std::data(outData_);
    const double* pWeights = weights_ ? std::data(*weights_) : nullptr;
    double* pAdjOutData = std::data(adjOutData);
    double* pAdjWeights = std::data(adjWeights);
    double* pF = std::data(F);

    vector<unique_ptr<BasePredictor>> basePredictors(iterationCount);

    for (size_t k = 0; k != iterationCount; ++k) {

        double absAdjOutDataSum = 0.0;

        // Visual C++ does autovectorize the following two loops

        if (pWeights == nullptr) {
            for (size_t i = 0; i < sampleCount; ++i) {
                double x = std::exp(-pF[i] * pOutData[i]);
                double z = pOutData[i] * (x + 1.0);
                pAdjOutData[i] = z;
                absAdjOutDataSum += std::abs(z);
                double u = x / square(x + 1.0);
                pAdjWeights[i] = u;
            }
        }
        else {
            for (size_t i = 0; i < sampleCount; ++i) {
                double x = std::exp(-pF[i] * pOutData[i]);
                double z = pOutData[i] * (x + 1.0);
                pAdjOutData[i] = z;
                absAdjOutDataSum += std::abs(z);
                double u = x / square(x + 1.0);
                u *= pWeights[i];
                pAdjWeights[i] = u;
            }
        }

        if (!(absAdjOutDataSum == numeric_limits<double>::infinity()))      // carefully written to trap NaN
            overflow_(opt);

        unique_ptr<BasePredictor> basePred = baseTrainer_->train(adjOutData, adjWeights, opt, threadCount);
        basePred->predict(inData_, eta, F);
        basePredictors[k] = move(basePred);
    }

    return std::make_shared<BoostPredictor>(variableCount_, globaLogOddsRatio_, eta, move(basePredictors));
}

//......................................................................................................................

shared_ptr<BoostPredictor> BoostTrainer::trainRegularizedLogit_(const BoostOptions& opt, size_t threadCount) const
{
    const size_t sampleCount = sampleCount_;
    const size_t iterationCount = opt.iterationCount();
    const double eta = opt.eta();
    const double gamma = opt.gamma();

    ArrayXd adjOutData(sampleCount);
    ArrayXd adjWeights(sampleCount);
    ArrayXd F = ArrayXd::Constant(sampleCount, globaLogOddsRatio_ / (gamma + 1.0));

    const double* pOutData = std::data(outData_);
    const double* pWeights = weights_ ? std::data(*weights_) : nullptr;
    double* pAdjOutData = std::data(adjOutData);
    double* pAdjWeights = std::data(adjWeights);
    double* pF = std::data(F);

    vector<unique_ptr<BasePredictor>> basePredictors(iterationCount);

    for (size_t k = 0; k != iterationCount; ++k) {

        double adjWeightSum = 0.0;
 
        // Visual C++ does autovectorize the following two loops

        if (pWeights == nullptr) {
            for (size_t i = 0; i < sampleCount; ++i) {
                double x = std::exp(-pF[i] * pOutData[i]);
                double z = pOutData[i] * (x + 1.0) / (gamma * x + 1.0);
                pAdjOutData[i] = z;
                double u = x * (gamma * x + 1.0) * std::pow(x + 1.0, gamma - 2.0);
                pAdjWeights[i] = u;
                adjWeightSum += u;
            }
        }
        else {
            for (size_t i = 0; i < sampleCount; ++i) {
                double x = std::exp(-pF[i] * pOutData[i]);
                double z = pOutData[i] * (x + 1.0) / (gamma * x + 1.0);
                pAdjOutData[i] = z;
                double u = x * (gamma * x + 1.0) * std::pow(x + 1.0, gamma - 2.0);
                u *= pWeights[i];
                pAdjWeights[i] = u;
                adjWeightSum += u;
            }
        }

        if (!(adjWeightSum < numeric_limits<double>::infinity()))   // carefully written to trap NaN
            overflow_(opt);

        unique_ptr<BasePredictor> basePred = baseTrainer_->train(adjOutData, adjWeights, opt, threadCount);
        basePred->predict(inData_, eta, F);
        basePredictors[k] = move(basePred);
    }

    return std::make_shared<BoostPredictor>(variableCount_, globaLogOddsRatio_, (1.0 + gamma) * eta, move(basePredictors));
}

//......................................................................................................................

void BoostTrainer::overflow_ [[noreturn]] (const BoostOptions& opt)
{
    double gamma = opt.gamma();
    string msg = (gamma == 1.0)
        ? "Numerical overflow in the boost algorithm.\nTry decreasing eta."
        : "Numerical overflow in the boost algorithm.\nTry decreasing eta or increasing gamma.";
    throw std::overflow_error(msg);
}

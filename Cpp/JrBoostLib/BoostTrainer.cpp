//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostTrainer.h"

#include "BasePredictor.h"
#include "BoostOptions.h"
#include "BoostPredictor.h"
#include "FastExp.h"
#include "TreeTrainer.h"


BoostTrainer::BoostTrainer(ArrayXXfc inData, ArrayXs outData, optional<ArrayXd> weights) :
    inData_{ std::move(inData) },
    sampleCount_{ static_cast<size_t>(inData_.rows()) },
    variableCount_{ static_cast<size_t>(inData_.cols()) },
    rawOutData_{ std::move(outData) },
    outData_{ 2.0 * rawOutData_.cast<double>() - 1.0 },
    weights_{ std::move(weights) },
    lor0_{ calculateLor0_() },
    baseTrainer_{ std::make_unique<TreeTrainer>(inData_, rawOutData_) }
{
    if (sampleCount_ == 0)
        throw std::invalid_argument("Train indata has 0 samples.");
    if (variableCount_ == 0)
        throw std::invalid_argument("Train indata has 0 variables.");
    if (!(inData_.abs() < numeric_limits<float>::infinity()).all())
        throw std::invalid_argument("Train indata has values that are infinity or NaN.");

    if (static_cast<size_t>(rawOutData_.rows()) != sampleCount_)
        throw std::invalid_argument("Train indata and outdata have different numbers of samples.");
    if ((rawOutData_ > 1).any())
        throw std::invalid_argument("Train outdata has values that are not 0 or 1.");

    if (weights_) {
        if (static_cast<size_t>(weights_->rows()) != sampleCount_)
            throw std::invalid_argument("Train indata and weights have different numbers of samples.");
        if (!(weights_->abs() < numeric_limits<double>::infinity()).all())
            throw std::invalid_argument("Train weights have values that are infinity or NaN.");
        if ((*weights_ <= 0.0).any())
            throw std::invalid_argument("Train weights have non-positive values.");
    }
}

BoostTrainer::~BoostTrainer() = default;


double BoostTrainer::calculateLor0_() const
{
    if (weights_)
        return  (
            std::log((rawOutData_.cast<double>() * (*weights_)).sum())
            - std::log(((1 - rawOutData_).cast<double>() * (*weights_)).sum())
        );
    else
        return  (
            std::log(static_cast<double>(rawOutData_.sum()))
            - std::log(static_cast<double>((1 - rawOutData_).sum()))
        );
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

    ArrayXd F = ArrayXd::Constant(sampleCount, lor0_ / 2.0);
    ArrayXd adjWeights(sampleCount);

    const double* pWeights = weights_ ? std::data(*weights_) : nullptr;
    const double* pF = &F(0);
    const double* pOutData = &outData_(0);
    double* pAdjWeights = &adjWeights(0);

    vector<unique_ptr<BasePredictor>> basePredictors(iterationCount);

    for (size_t i = 0; i != iterationCount; ++i) {

        double m = 0.0;

        if (!opt.fastExp()) {

            if (pWeights == nullptr) {
                // this loop should be auto-vectorized
                for (size_t j = 0; j != sampleCount; ++j) {
                    const double x = std::exp(-pF[j] * pOutData[j]);
                    pAdjWeights[j] = x;
                    m = (x > m) ? x : m;
                }
            }
            else {
                // this loop should be auto-vectorized
                for (size_t j = 0; j != sampleCount; ++j) {
                    const double x = pWeights[j] * std::exp(-pF[j] * pOutData[j]);
                    pAdjWeights[j] = x;
                    m = (x > m) ? x : m;
                }
            }
        }

        else {

            size_t j = 0;

 #if defined(__AVX512F__) && defined(__AVX512DQ__)

            // WARNING: NOT TESTED!

            __m512d m8 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

            constexpr __m512d negZero8 = { -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0 };
            if (pWeights == nullptr) {
                while (j + 8 <= sampleCount) {
                    const __m512d F8 = _mm512_load_pd(pF + j);
                    const __m512d y8 = _mm512_load_pd(pOutData + j);
                    __m512d x8 = _mm512_mul_pd(F8, y8);
                    x8 = _mm512_xor_pd(x8, negZero8);       // x8 = -x8
                    x8 = fastExp(x8);
                    _mm512_store_pd(pAdjWeights + j, x8);
                    m8 = _mm512_max_pd(m8, x8);
                    j += 8;
                }
            }
            else {
                while (j + 8 <= sampleCount) {
                    const __m512d F8 = _mm512_load_pd(pF + j);
                    const __m512d y8 = _mm512_load_pd(pOutData + j);
                    const __m512d w8 = _mm512_loadu_pd(pWeights + j);     // allocated by client, alignment unknown
                    __m512d x8 = _mm512_mul_pd(F8, y8);
                    x8 = _mm512_xor_pd(x8, negZero8);       // x8 = -x8
                    x8 = fastExp(x8);
                    x8 = _mm512_mul_pd(x8, w8);
                    _mm512_store_pd(pAdjWeights + j, x8);
                    m8 = _mm512_max_pd(m8, x8);
                    j += 8;
                }
            }

            m = *std::max_element(std::begin(m8.m512d_f64), std::end(m8.m512d_f64));

#elif defined(__AVX2__)

            __m256d m4 = { 0.0, 0.0, 0.0, 0.0 };

            constexpr __m256d negZero4 = { -0.0, -0.0, -0.0, -0.0 };
            if (pWeights == nullptr) {
                while (j + 4 <= sampleCount) {
                    const  __m256d F4 = _mm256_load_pd(pF + j);
                    const __m256d y4 = _mm256_load_pd(pOutData + j);
                    __m256d x4 = _mm256_mul_pd(F4, y4);
                    x4 = _mm256_xor_pd(x4, negZero4);       // x4 = -x4
                    x4 = fastExp(x4);
                    _mm256_store_pd(pAdjWeights + j, x4);
                    m4 = _mm256_max_pd(m4, x4);
                    j += 4;
                }
            }
            else {
                while (j + 4 <= sampleCount) {
                    const __m256d F4 = _mm256_load_pd(pF + j);
                    const __m256d y4 = _mm256_load_pd(pOutData + j);
                    const __m256d w4 = _mm256_loadu_pd(pWeights + j);     // allocated by client, alignment unknown
                    __m256d x4 = _mm256_mul_pd(F4, y4);
                    x4 = _mm256_xor_pd(x4, negZero4);       // x4 = -x4
                    x4 = fastExp(x4);
                    x4 = _mm256_mul_pd(x4, w4);
                    _mm256_store_pd(pAdjWeights + j, x4);
                    m4 = _mm256_max_pd(m4, x4);
                    j += 4;
                }
            }

            m = *std::max_element(std::begin(m4.m256d_f64), std::end(m4.m256d_f64));
#endif
            if (pWeights == nullptr) {
                // this loop should not be auto-vectorized
                while (j < sampleCount) {
                    const double x = fastExp(-pF[j] * pOutData[j]);
                    pAdjWeights[j] = x;
                    m = (x > m) ? x : m;
                    ++j;
                }
            }
            else {
                // this loop should not  be auto-vectorized
                while (j < sampleCount) {
                    const double x = pWeights[j] * fastExp(-pF[j] * pOutData[j]);
                    pAdjWeights[j] = x;
                    m = (x > m) ? x : m;
                    ++j;
                }
            }

        }  // end if (!opt.fastExp())

        if (m == numeric_limits<double>::infinity())
            overflow_(opt);

        unique_ptr<BasePredictor> basePred = baseTrainer_->train(outData_, adjWeights, opt, threadCount);
        basePred->predict(inData_, eta, F);
        basePredictors[i] = move(basePred);
    }

    return std::make_shared<BoostPredictor>(variableCount_, lor0_, 2 * eta, move(basePredictors));
}

//......................................................................................................................

shared_ptr<BoostPredictor> BoostTrainer::trainLogit_(const BoostOptions& opt, size_t threadCount) const
{
    const size_t sampleCount = sampleCount_;

    const size_t iterationCount = opt.iterationCount();
    const double eta = opt.eta();

    ArrayXd F = ArrayXd::Constant(sampleCount, lor0_);
    ArrayXd adjOutData(sampleCount);
    ArrayXd adjWeights(sampleCount);

    vector<unique_ptr<BasePredictor>> basePredictors(iterationCount);

    for (size_t i = 0; i != iterationCount; ++i) {

        //ArrayXd x = (-F * outData_).exp();               
        //adjOutData = outData_ * (x + 1.0);
        //adjWeights = x / (x + 1.0).square()

        const double* pF = &F(0);
        const double* pOutData = &outData_(0);
        double* pAdjOutData = &adjOutData(0);
        double* pAdjWeights = &adjWeights(0);

        for (size_t j = 0; j != sampleCount; ++j) {
            double x = std::exp(-pF[j] * pOutData[j]);
            pAdjOutData[j] = pOutData[j] * (x + 1.0);
            pAdjWeights[j] = x / ((x + 1.0) * (x + 1.0));
        }

        if (weights_)
            adjWeights *= (*weights_);

        if (adjWeights.maxCoeff() == numeric_limits<double>::infinity())
            overflow_(opt);

        unique_ptr<BasePredictor> basePred = baseTrainer_->train(adjOutData, adjWeights, opt, threadCount);
        basePred->predict(inData_, eta, F);
        basePredictors[i] = move(basePred);
    }

    return std::make_shared<BoostPredictor>(variableCount_, lor0_, eta, move(basePredictors));
}

//......................................................................................................................

shared_ptr<BoostPredictor> BoostTrainer::trainRegularizedLogit_(const BoostOptions& opt, size_t threadCount) const
{
    const size_t sampleCount = sampleCount_;

    const double gamma = opt.gamma();
    const size_t iterationCount = opt.iterationCount();
    const double eta = opt.eta();

    ArrayXd F = ArrayXd::Constant(sampleCount, lor0_ / (gamma + 1.0));
    ArrayXd adjOutData(sampleCount);
    ArrayXd adjWeights(sampleCount);

    vector<unique_ptr<BasePredictor>> basePredictors(iterationCount);

    for (size_t i = 0; i != iterationCount; ++i) {

        //ArrayXd x = (-F * outData_).exp();
        //adjOutData = outData_ * (x + 1.0) / (gamma * x + 1.0);
        //adjWeights = x * (gamma * x + 1.0) * (x + 1.0).pow(gamma - 2.0);
 
        const double* pF = &F(0);
        const double* pOutData = &outData_(0);
        double* pAdjOutData = &adjOutData(0);
        double* pAdjWeights = &adjWeights(0);

        for (size_t j = 0; j != sampleCount; ++j) {
            double x = std::exp(-pF[j] * pOutData[j]);
            pAdjOutData[j] = pOutData[j] * (x + 1.0) / (gamma * x + 1.0);
            pAdjWeights[j] = x * (gamma * x + 1.0) * std::pow(x + 1.0, gamma - 2.0);
        }

        if (weights_)
            adjWeights *= (*weights_);

        if (adjWeights.maxCoeff() == numeric_limits<double>::infinity())
            overflow_(opt);

        unique_ptr<BasePredictor> basePred = baseTrainer_->train(adjOutData, adjWeights, opt, threadCount);
        basePred->predict(inData_, eta, F);
        basePredictors[i] = move(basePred);
    }

    return std::make_shared<BoostPredictor>(variableCount_, lor0_, (1.0 + gamma) * eta, move(basePredictors));
}

//......................................................................................................................

void BoostTrainer::overflow_ [[noreturn]] (const BoostOptions& opt)
{
    double gamma = opt.gamma();
    string msg = gamma == 1.0
        ? "Numerical overflow in the boost algorithm.\nTry decreasing eta."
        : "Numerical overflow in the boost algorithm.\nTry decreasing eta or increasing gamma.";
    throw std::overflow_error(msg);
}

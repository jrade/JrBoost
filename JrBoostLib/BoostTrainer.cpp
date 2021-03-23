//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostTrainer.h"
#include "BoostOptions.h"
#include "BoostPredictor.h"
#include "StumpTrainer.h"
#include "SimplePredictor.h"
#include "FastMath.h"
#include "InterruptHandler.h"
#include "SortedIndices.h"


BoostTrainer::BoostTrainer(ArrayXXf inData, ArrayXs outData, optional<ArrayXd> weights) :
    inData_{ std::move(inData) },
    sampleCount_{ static_cast<size_t>(inData_.rows()) },
    variableCount_{ static_cast<size_t>(inData_.cols()) },
    rawOutData_{ std::move(outData) },
    outData_{ 2.0 * rawOutData_.cast<double>() - 1.0 },
    weights_{ std::move(weights) },
    lor0_{ calculateLor0_() },
    baseTrainer_{ std::make_shared<StumpTrainer>(inData_, rawOutData_) }
{
    ASSERT(inData_.rows() != 0);
    ASSERT(inData_.cols() != 0);
    ASSERT(outData_.rows() == inData_.rows());

    ASSERT((inData_ > -numeric_limits<float>::infinity()).all());
    ASSERT((inData_ < numeric_limits<float>::infinity()).all());
    ASSERT((rawOutData_ < 2).all());
}

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

unique_ptr<BoostPredictor> BoostTrainer::train(const BoostOptions& opt) const
{
    PROFILE::PUSH(PROFILE::BOOST_TRAIN);

    unique_ptr<BoostPredictor> pred;
    switch (opt.method()) {
    case BoostOptions::Ada:
        pred = trainAda_(opt);
        break;
    case BoostOptions::Alpha:
        pred = trainAlpha_(opt);
        break;
    default:
        ASSERT(false);
    }

    const size_t ITEM_COUNT = sampleCount_ *  opt.iterationCount();
    PROFILE::POP(ITEM_COUNT);

    return pred;
}

unique_ptr<BoostPredictor> BoostTrainer::trainAda_(BoostOptions opt) const
{
    const size_t sampleCount = sampleCount_;

    const size_t iterationCount = opt.iterationCount();
    const double eta = opt.eta();
    const double minAbsSampleWeight = opt.minAbsSampleWeight();
    const double minRelSampleWeight = opt.minRelSampleWeight();
    const bool useFastExp = opt.fastExp();

    ArrayXd F = ArrayXd::Constant(sampleCount, lor0_ / 2.0);
    ArrayXd adjWeights(sampleCount);

    vector<unique_ptr<SimplePredictor>> basePredictors(iterationCount);
    vector<double> coeff(iterationCount);

    for (size_t i = 0; i < iterationCount; ++i) {

        if (useFastExp) {

            // 1. Eigen
            /*{
                adjWeights = fastExp(-F * outData_);
            }*/

            // 2. Scalar loop (not vectorized with MSVS 2019)
            {
                const double* pF = &F(0);
                const double* pOutData = &outData_(0);
                double* pAdjWeights = &adjWeights(0);

                for (size_t j = 0; j < sampleCount; ++j)
                    pAdjWeights[j] = fastExp(-pF[j] * pOutData[j]);
            }
            

            // 3. SIMD loop
            /*{
                size_t j = 0;

                for (; j <= sampleCount - 8; j += 8)
                {
                    vcl::Vec8d vF, vOutData;
                    vF.load(&F(j));
                    vOutData.load(&outData_(j));
                    vcl::Vec8d vAdjWeight = fastExp(-vF * vOutData);
                    vAdjWeight.store(&adjWeights(j));
                }

                int k = static_cast<int>(sampleCount - j);
                vcl::Vec8d vF, vOutData;
                vF.load_partial(k, &F(j));
                vOutData.load_partial(k, &outData_(j));
                vcl::Vec8d vAdjWeight = fastExp(-vF * vOutData);
                vAdjWeight.store_partial(k, &adjWeights(j));
            }*/
        }

        else {

            // 1. Eigen
            /*{
                adjWeights = (-F * outData_).exp();
            }*/

            // 2. Scalar loop (vectorized with MSVS 2019)
            {
                const double* pF = &F(0);
                const double* pOutData = &outData_(0);
                double* pAdjWeights = &adjWeights(0);

                for (size_t j = 0; j < sampleCount; ++j)
                    pAdjWeights[j] = std::exp(-pF[j] * pOutData[j]);
            }            

            // 3. SIMD loop
            /*{
                size_t j = 0;

                for (; j <= sampleCount - 8; j += 8)
                {
                    vcl::Vec8d vF, vOutData;
                    vF.load(&F(j));
                    vOutData.load(&outData_(j));
                    vcl::Vec8d vAdjWeight = vcl::exp(-vF * vOutData);
                    vAdjWeight.store(&adjWeights(j));
                }

                int k = static_cast<int>(sampleCount - j);
                vcl::Vec8d vF, vOutData;
                vF.load_partial(k, &F(j));
                vOutData.load_partial(k, &outData_(j));
                vcl::Vec8d vAdjWeight = vcl::exp(-vF * vOutData);
                vAdjWeight.store_partial(k, &adjWeights(j));
            }*/
        }

        if (weights_)
            adjWeights *= (*weights_);

        double minSampleWeight = minAbsSampleWeight;
        if (minRelSampleWeight > 0.0)
            minSampleWeight = std::max(minSampleWeight, adjWeights.maxCoeff() * minRelSampleWeight);
        opt.setMinSampleWeight(minSampleWeight);

        unique_ptr<SimplePredictor> basePred = baseTrainer_->train(outData_, adjWeights, opt);
        basePred->predict(inData_, eta, F);
        basePredictors[i] = std::move(basePred);
        coeff[i] = 2.0 * eta;
    }

    return unique_ptr<BoostPredictor>(
        new BoostPredictor(variableCount_, lor0_, std::move(coeff), std::move(basePredictors))
    );
}

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<BoostPredictor> BoostTrainer::trainAlpha_(BoostOptions opt) const
{
    const size_t sampleCount = sampleCount_;

    const double alpha = opt.alpha();
    const size_t iterationCount = opt.iterationCount();
    const double eta = opt.eta();
    const double minAbsSampleWeight = opt.minAbsSampleWeight();
    const double minRelSampleWeight = opt.minRelSampleWeight();
    const bool useFastExp = opt.fastExp();

    ArrayXd F = ArrayXd::Constant(sampleCount, lor0_ / (alpha + 1.0));
    ArrayXd adjOutData(sampleCount);
    ArrayXd adjWeights(sampleCount);

    vector<unique_ptr<SimplePredictor>> basePredictors(iterationCount);
    vector<double> coeff(iterationCount);

    for (size_t i = 0; i < iterationCount; ++i) {

        if(useFastExp) {

            // 1. Eigen    [37]
            /*{
                ArrayXd x = fastExp(-F * outData_);                   
                adjOutData = outData_ * (x + 1.0) / (alpha * x + 1.0);
                adjWeights = x * (alpha * x + 1.0) * fastPow(x + 1.0, alpha - 2.0);
            }*/

            // 2. Scalar loop (not vectorized by VS 2019)   [34]
            {
                const double* pF = &F(0);
                const double* pOutData = &outData_(0);
                double* pAdjOutData = &adjOutData(0);
                double* pAdjWeights = &adjWeights(0);

                for (size_t j = 0; j < sampleCount; ++j) {
                    double x = fastExp(-pF[j] * pOutData[j]);
                    pAdjOutData[j] = pOutData[j] * (x + 1.0) / (alpha * x + 1.0);
                    pAdjWeights[j] = x * (alpha * x + 1.0) * fastPow(x + 1.0, alpha - 2.0);
                }
            }
            
            // 3. SIMD loop    [29]
            /*{
                size_t j = 0;

                for (; j <= sampleCount - 4; j += 4)
                {
                    vcl::Vec4d vF, vOutData;
                    vF.load(&F(j));
                    vOutData.load(&outData_(j));
                    vcl::Vec4d x = fastExp(-vOutData * vF);
                    vcl::Vec4d vAdjOutData = vOutData * (x + 1.0) / (alpha * x + 1.0);
                    vcl::Vec4d vAdjWeights = x * (alpha * x + 1.0) * fastPow(x + 1.0, alpha - 2.0);
                    vAdjOutData.store(&adjOutData(j));
                    vAdjWeights.store(&adjWeights(j));
                }

                int k = static_cast<int>(sampleCount - j);
                vcl::Vec4d vF, vOutData;
                vF.load_partial(k, &F(j));
                vOutData.load_partial(k, &outData_(j));
                vcl::Vec4d x = fastExp(-vOutData * vF);
                vcl::Vec4d vAdjOutData = vOutData * (x + 1.0) / (alpha * x + 1.0);
                vcl::Vec4d vAdjWeights = x * (alpha * x + 1.0) * fastPow(x + 1.0, alpha - 2.0);
                vAdjOutData.store_partial(k, &adjOutData(j));
                vAdjWeights.store_partial(k, &adjWeights(j));
            }*/
        }

        else {

            // 1. Eigen    [150]
            /*{
                ArrayXd x = (-F * outData_).exp();
                adjOutData = outData_ * (x + 1.0) / (alpha * x + 1.0);
                adjWeights = x * (alpha * x + 1.0) * (x + 1.0).pow(alpha - 2.0);
            }*/
 
            // 2. Scalar loop (vectorized by VS 2019)      [89]
            {
                const double* pF = &F(0);
                const double* pOutData = &outData_(0);
                double* pAdjOutData = &adjOutData(0);
                double* pAdjWeights = &adjWeights(0);

                for (size_t j = 0; j < sampleCount; ++j) {
                    double x = std::exp(-pF[j] * pOutData[j]);
                    pAdjOutData[j] = pOutData[j] * (x + 1.0) / (alpha * x + 1.0);
                    pAdjWeights[j] = x * (alpha * x + 1.0) * std::pow(x + 1.0, alpha - 2.0);
                }
            }

            // 3. SIMD loop        [113]
            /*{
                size_t j = 0;

                for (; j <= sampleCount - 4; j += 4) {
                    vcl::Vec4d vF, vOutData;
                    vF.load(&F(j));
                    vOutData.load(&outData_(j));
                    vcl::Vec4d x = vcl::exp(-vOutData * vF);
                    vcl::Vec4d vAdjOutData = vOutData * (x + 1.0) / (alpha * x + 1.0);
                    vcl::Vec4d vAdjWeights = x * (alpha * x + 1.0) * vcl::pow(x + 1.0, alpha - 2.0);
                    vAdjOutData.store(&adjOutData(j));
                    vAdjWeights.store(&adjWeights(j));
                }

                int k = static_cast<int>(sampleCount - j);
                vcl::Vec4d vF, vOutData;
                vF.load_partial(k, &F(j));
                vOutData.load_partial(k, &outData_(j));
                vcl::Vec4d x = vcl::exp(-vOutData * vF);
                vcl::Vec4d vAdjOutData = vOutData * (x + 1.0) / (alpha * x + 1.0);
                vcl::Vec4d vAdjWeights = x * (alpha * x + 1.0) * vcl::pow(x + 1.0, alpha - 2.0);
                vAdjOutData.store_partial(k, &adjOutData(j));
                vAdjWeights.store_partial(k, &adjWeights(j));
            }*/
        }

        if (weights_)
            adjWeights *= (*weights_);

        double minSampleWeight = minAbsSampleWeight;
        if (minRelSampleWeight > 0.0)
            minSampleWeight = std::max(minSampleWeight, adjWeights.maxCoeff() * minRelSampleWeight);
        opt.setMinSampleWeight(minSampleWeight);

        unique_ptr<SimplePredictor> basePred = baseTrainer_->train(adjOutData, adjWeights, opt);
        basePred->predict(inData_, eta, F);
        basePredictors[i] = std::move(basePred);
        coeff[i] = (1.0 + alpha) * eta;
    }

    return unique_ptr<BoostPredictor>(
        new BoostPredictor(variableCount_, lor0_, std::move(coeff), std::move(basePredictors))
    );
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXd BoostTrainer::trainAndEval(
    CRefXXf testInData,
    CRefXs testOutData,
    const vector<BoostOptions>& opt,
    function<Array3d(CRefXs, CRefXd)> lossFun
) const
{
    ASSERT(testInData.rows() == testOutData.rows());
    size_t optCount = opt.size();

    // In each iteration of the loop we build and evaluate a classifier with one set of options
    // Differents sets of options can take very different time.
    // To ensure that the OMP threads are balanced we use dynamical scheduling and we sort the options objects
    // from the most time-consuming to the least.
    // In one test case, this decraesed the time spent waiting at the barrier
    // from 3.9% to 0.2% of the total execution time.

    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(
        cbegin(opt), 
        cend(opt),
        begin(optIndicesSortedByCost),
        [](const auto& opt) { return -opt.cost(); }
    );

    ArrayXd scores(optCount);
    std::exception_ptr ep;
    atomic<bool> exceptionThrown = false;

    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();

        #pragma omp for nowait schedule(dynamic)
        for (int i = 0; i < static_cast<int>(optCount); ++i) {

            if (exceptionThrown) continue;

            try {
                if (threadId == 0 && currentInterruptHandler != nullptr)
                    currentInterruptHandler->check();  // throws if there is a keyboard interrupt

                size_t j = optIndicesSortedByCost[i];
                unique_ptr<BoostPredictor> pred = train(opt[j]);
                ArrayXd predData = pred->predict(testInData);
                scores(j) = lossFun(testOutData, predData)(2);
            }

            catch (const std::exception&) {
                #pragma omp critical
                if (!exceptionThrown) {
                    ep = std::current_exception();
                    exceptionThrown = true;
                }
            }

        } // don't wait here ...
        PROFILE::PUSH(PROFILE::OMP_BARRIER);

    } // ... but here so we can measure the wait time
    PROFILE::POP();

    if (exceptionThrown) std::rethrow_exception(ep);

    return scores;
}

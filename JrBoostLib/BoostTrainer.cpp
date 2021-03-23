//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostTrainer.h"
#include "BoostOptions.h"
#include "BoostPredictor.h"
#include "StumpTrainer.h"
#include "SimplePredictor.h"
#include "SortedIndices.h"
#include "InterruptHandler.h"


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
            log((rawOutData_.cast<double>() * (*weights_)).sum())
            - log(((1 - rawOutData_).cast<double>() * (*weights_)).sum())
        );
    else
        return  (
            log(static_cast<double>(rawOutData_.sum()))
            - log(static_cast<double>((1 - rawOutData_).sum()))
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

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<BoostPredictor> BoostTrainer::trainAda_(BoostOptions opt) const
{
    const size_t iterationCount = opt.iterationCount();
    const double eta = opt.eta();
    const double minAbsSampleWeight = opt.minAbsSampleWeight();
    const double minRelSampleWeight = opt.minRelSampleWeight();
    const bool fastExp = opt.fastExp();

    ArrayXd F = ArrayXd::Constant(sampleCount_, lor0_ / 2.0);
    ArrayXd adjWeights(sampleCount_);

    vector<unique_ptr<SimplePredictor>> basePredictors(iterationCount);
    vector<double> coeff(iterationCount);

    for (size_t i = 0; i < iterationCount; ++i) {
        adjWeights = -F * outData_;
        const double a = adjWeights.maxCoeff();

        if (fastExp) {
            constexpr double c1 = (1ll << 52) / 0.6931471805599453;
            constexpr double c2 = (1ll << 52) * (1023 - 0.04367744890362246);
            reinterpret_cast<ArrayXs&>(adjWeights) = (c1 * adjWeights + (c2 - c1 * a)).cast<uint64_t>();
        }
        else
            adjWeights = (adjWeights - a).exp();

        if (weights_) 
            adjWeights *= (*weights_);

        double minSampleWeight = 0.0;
        if (minAbsSampleWeight > 0.0)
            minSampleWeight = exp(-a) * minAbsSampleWeight;
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
    const double alpha = opt.alpha();
    const size_t iterationCount = opt.iterationCount();
    const double eta = opt.eta();
    const double minAbsSampleWeight = opt.minAbsSampleWeight();
    const double minRelSampleWeight = opt.minRelSampleWeight();
    const bool fastExp = opt.fastExp();

    ArrayXd F = ArrayXd::Constant(sampleCount_, lor0_ / (alpha + 1.0));
    ArrayXd adjOutData(sampleCount_);
    ArrayXd adjWeights(sampleCount_);

    vector<unique_ptr<SimplePredictor>> basePredictors(iterationCount);
    vector<double> coeff(iterationCount);

    ArrayXd tmp00(sampleCount_);
    ArrayXd tmp0(sampleCount_);
    ArrayXd tmpAlpha(sampleCount_);
    ArrayXd tmp1(sampleCount_);
    ArrayXd tmp2(sampleCount_);

    for (size_t i = 0; i < iterationCount; ++i) {

        tmp00 = -F * outData_;
        const double a = tmp00.maxCoeff();
        tmp00 -= a;

        if (fastExp) {

            constexpr double c1 = (1ll << 52) / 0.6931471805599453;
            constexpr double c2 = (1ll << 52) * (1023 - 0.04367744890362246);

            // tmp00 approximately = tmp00.exp()
            reinterpret_cast<ArrayXs&>(tmp00) = (c1 * tmp00 + c2).cast<uint64_t>();

            tmp0 = exp(a) * tmp00;          // exp(-F * outData_)
            tmpAlpha = tmp0 + alpha;
            tmp1 = tmp0 + 1.0;

            // tmp2 approximately = tmp1.pow(alpha)
            reinterpret_cast<ArrayXs&>(tmp2)
                = (alpha * reinterpret_cast<ArrayXs&>(tmp1).cast<double>() + (1.0 - alpha) * c2).cast<uint64_t>();

            adjOutData = outData_ * tmp1 / tmpAlpha;
            adjWeights = tmp00 * (tmp2 / tmp1.square()) * tmpAlpha;
        }
        else {
            tmp00 = tmp00.exp();      // normalized exp(-F * outData_)

            tmp0 = exp(a) * tmp00;          // exp(-F * outData_)
            tmpAlpha = tmp0 + alpha;
            tmp1 = tmp0 + 1.0;

            adjOutData = outData_ * tmp1 / tmpAlpha;
            adjWeights = tmp00 * tmp1.pow(alpha - 2.0) * tmpAlpha;
        }

        if (weights_)
            adjWeights *= (*weights_);

        double minSampleWeight = 0.0;
        if (minAbsSampleWeight > 0.0)
            minSampleWeight = exp(-a) * minAbsSampleWeight;
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

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
    f0_{ calculateF0_() },
    baseTrainer_{ std::make_shared<StumpTrainer>(inData_, rawOutData_) }
{
    ASSERT(inData_.rows() != 0);
    ASSERT(inData_.cols() != 0);
    ASSERT(outData_.rows() == inData_.rows());

    ASSERT((inData_ > -numeric_limits<float>::infinity()).all());
    ASSERT((inData_ < numeric_limits<float>::infinity()).all());
    ASSERT((rawOutData_ < 2).all());
}

double BoostTrainer::calculateF0_() const
{
    if (weights_)
        return  (
            log((rawOutData_.cast<double>() * (*weights_)).sum())
            - log(((1 - rawOutData_).cast<double>() * (*weights_)).sum())
        ) / 2.0;
    else
        return  (
            log(static_cast<double>(rawOutData_.sum()))
            - log(static_cast<double>((1 - rawOutData_).sum()))
        ) / 2.0;
}

unique_ptr<BoostPredictor> BoostTrainer::train(const BoostOptions& opt) const
{
    PROFILE::PUSH(PROFILE::BOOST_TRAIN);

    auto pred = opt.method() == BoostOptions::Method::Ada ? trainAda_(opt) : trainLogit_(opt);

    const size_t ITEM_COUNT = sampleCount_ *  opt.iterationCount();
    PROFILE::POP(ITEM_COUNT);

    return pred;
}

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<BoostPredictor> BoostTrainer::trainAda_(const BoostOptions& opt) const
{
    const double eta = opt.eta();
    const size_t iterationCount = opt.iterationCount();

    F_.resize(sampleCount_);
    F_ = f0_;

    vector<unique_ptr<SimplePredictor>> basePredictors(iterationCount);
    vector<double> coeff(iterationCount);

    for (size_t i = 0; i < iterationCount; ++i) {
        adjWeights_ = F_ * outData_;
        adjWeights_ = (adjWeights_.minCoeff() - adjWeights_).exp();
        if (weights_) 
            adjWeights_ *= (*weights_);
        unique_ptr<SimplePredictor> basePred = baseTrainer_->train(outData_, adjWeights_, opt);
        basePred->predict(inData_, eta, F_);
        basePredictors[i] = std::move(basePred);
        coeff[i] = 2.0 * eta;
    }

    return unique_ptr<BoostPredictor>(
        new BoostPredictor(variableCount_, 2.0 * f0_, std::move(coeff), std::move(basePredictors))
    );
}

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<BoostPredictor> BoostTrainer::trainLogit_(const BoostOptions& opt) const
{
    const size_t logStep = 0;
    const size_t sampleCount = inData_.rows();
    const size_t variableCount = inData_.cols();

    ArrayXd F = ArrayXd::Constant(sampleCount, f0_);

    ArrayXd adjOutData, adjWeights, f;
    vector<unique_ptr<SimplePredictor>> basePredictors;
    vector<double> coeff;

    const double eta = opt.eta();
    const size_t n = opt.iterationCount();
    for (size_t i = 0; i < n; ++i) {

        double FAbsMin = F.abs().minCoeff();
        adjWeights = 1.0 / ((F - FAbsMin).exp() + (-F - FAbsMin).exp()).square();
        if (weights_)
            adjWeights_ *= (*weights_);
        adjWeights /= adjWeights.maxCoeff();

        adjOutData = outData_ * (1.0 + (-2.0 * outData_ * F).exp()) / 2.0;

        unique_ptr<SimplePredictor > basePredictor = baseTrainer_->train(adjOutData, adjWeights, opt);

        //f = basePredictor->predict(inData_);
        double c = eta / std::max(0.5, f.abs().maxCoeff());
        F += c * f;
        coeff.push_back(2 * c);
        basePredictors.push_back(std::move(basePredictor));

        if constexpr(logStep > 0 && i % logStep == 0) {
            cout << i << "(" << eta << ")" << endl;
            cout << "Fy: " << (outData_ * F).minCoeff() << " - " << (outData_ * F).maxCoeff() << endl;
            cout << "w: " << adjWeights.minCoeff() << " - " << adjWeights.maxCoeff();
            cout << " -> " << 100.0 * (adjWeights != 0.0).cast<double>().sum() / sampleCount << "%" << endl;
            cout << "y*y: " << (outData_ * adjOutData).minCoeff() << " - " << (outData_ * adjOutData).maxCoeff() << endl;
            cout << "fy: " << (f * outData_).minCoeff() << " - " << (f * outData_).maxCoeff() << endl << endl;
        }
    }

    return unique_ptr<BoostPredictor>(
        new BoostPredictor(variableCount, 2 * f0_, std::move(coeff), std::move(basePredictors))
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

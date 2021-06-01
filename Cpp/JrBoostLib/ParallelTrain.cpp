
//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "ParallelTrain.h"
#include "BoostOptions.h"
#include "BoostPredictor.h"
#include "BoostTrainer.h"
#include "InterruptHandler.h"
#include "SortedIndices.h"


// In each iteration of the loop we build a classifier with one set of options
// Differents sets of options can take very different time.
// To ensure that the OMP threads are balanced we use dynamical scheduling and we sort the options objects
// from the most time-consuming to the least.

//----------------------------------------------------------------------------------------------------------------------

vector<shared_ptr<BoostPredictor>> parallelTrain(const BoostTrainer& trainer, const vector<BoostOptions>& opt)
{
    const size_t optCount = opt.size();

    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(
        cbegin(opt),
        cend(opt),
        begin(optIndicesSortedByCost),
        [](const auto& opt) { return -opt.cost(); }
    );

    vector<shared_ptr<BoostPredictor>> pred(optCount);
    std::exception_ptr ep;
    atomic<bool> exceptionThrown = false;

#pragma omp parallel
    {
#pragma omp for nowait schedule(dynamic)
        for (int i = 0; i < static_cast<int>(optCount); ++i) {

            if (exceptionThrown) continue;

            try {
                if (omp_get_thread_num() == 0 && currentInterruptHandler != nullptr)
                    currentInterruptHandler->check();  // throws if there is a keyboard interrupt

                size_t j = optIndicesSortedByCost[i];
                pred[j] = trainer.train(opt[j]);    // may also throw
            }

            catch (const std::exception&) {
#pragma omp critical
                if (!exceptionThrown) {
                    ep = std::current_exception();
                    exceptionThrown = true;
                }
            }

        } // don't wait here ...

        PROFILE::PUSH(PROFILE::THREAD_SYNCH);
    } // ... but here so we can measure the wait time
    PROFILE::POP();

    if (exceptionThrown) std::rethrow_exception(ep);

    return pred;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXXd parallelTrainAndPredict(const BoostTrainer& trainer, const vector<BoostOptions>& opt, CRefXXf testInData)
{
    if (testInData.rows() == 0)
        throw std::invalid_argument("Test indata has 0 samples.");

    const size_t sampleCount = testInData.rows();
    const size_t optCount = opt.size();

    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(
        cbegin(opt),
        cend(opt),
        begin(optIndicesSortedByCost),
        [](const auto& opt) { return -opt.cost(); }
    );

    ArrayXXd predData(sampleCount, optCount);
    std::exception_ptr ep;
    atomic<bool> exceptionThrown = false;

#pragma omp parallel
    {
#pragma omp for nowait schedule(dynamic)
        for (int i = 0; i < static_cast<int>(optCount); ++i) {

            if (exceptionThrown) continue;

            try {
                if (omp_get_thread_num() == 0 && currentInterruptHandler != nullptr)
                    currentInterruptHandler->check();  // throws if there is a keyboard interrupt

                size_t j = optIndicesSortedByCost[i];
                shared_ptr<BoostPredictor> pred = trainer.train(opt[j]);    // may also throw
                predData.col(j) = pred->predict(testInData);
            }

            catch (const std::exception&) {
#pragma omp critical
                if (!exceptionThrown) {
                    ep = std::current_exception();
                    exceptionThrown = true;
                }
            }

        } // don't wait here ...

        PROFILE::PUSH(PROFILE::THREAD_SYNCH);
    } // ... but here so we can measure the wait time
    PROFILE::POP();

    if (exceptionThrown) std::rethrow_exception(ep);

    return predData;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXd parallelTrainAndEval(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    CRefXXf testInData, CRefXs testOutData, function<double(CRefXs, CRefXd)> lossFun
)
{
    size_t optCount = opt.size();

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
#pragma omp for nowait schedule(dynamic)
        for (int i = 0; i < static_cast<int>(optCount); ++i) {

            if (exceptionThrown) continue;

            try {
                if (omp_get_thread_num() == 0 && currentInterruptHandler != nullptr)
                    currentInterruptHandler->check();  // throws if there is a keyboard interrupt

                size_t j = optIndicesSortedByCost[i];
                shared_ptr<BoostPredictor> pred = trainer.train(opt[j]);    // may also throw
                ArrayXd predData = pred->predict(testInData);
                scores(j) = lossFun(testOutData, predData);
            }

            catch (const std::exception&) {
#pragma omp critical
                if (!exceptionThrown) {
                    ep = std::current_exception();
                    exceptionThrown = true;
                }
            }

        } // don't wait here ...

        PROFILE::PUSH(PROFILE::THREAD_SYNCH);
    } // ... but here so we can measure the wait time
    PROFILE::POP();

    if (exceptionThrown) std::rethrow_exception(ep);

    return scores;
}

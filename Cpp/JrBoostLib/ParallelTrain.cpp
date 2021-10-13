//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "ParallelTrain.h"

#include "BoostOptions.h"
#include "BoostPredictor.h"
#include "BoostTrainer.h"
#include "ExceptionSafeOmp.h"
#include "SortedIndices.h"


vector<shared_ptr<BoostPredictor>> parallelTrain(const BoostTrainer& trainer, const vector<BoostOptions>& opt)
{
    const size_t optCount = size(opt);

    // In order to keep the threads balanced we will process the options one-by-one in order
    // from the most computaionally expensive to the least computationally expensive

    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(
        cbegin(opt),
        cend(opt),
        begin(optIndicesSortedByCost),
        [](const auto& opt) { return -opt.cost(); }
    );

    vector<shared_ptr<BoostPredictor>> pred(optCount);

    const size_t totalThreadCount = omp_get_max_threads();

    size_t outerThreadCount;
    if (::globParallelTree) {
        omp_set_nested(true);
        outerThreadCount = ::globOuterThreadCount;
        if (outerThreadCount == 0)
            // the square root of the total thread count is a reasonable default for the outer thread count
            outerThreadCount = static_cast<size_t>(std::round(std::sqrt(totalThreadCount)));
        if (outerThreadCount > totalThreadCount)
            outerThreadCount = totalThreadCount;
    }
    else
        outerThreadCount = totalThreadCount;
    if (outerThreadCount > optCount)
        outerThreadCount = optCount;

    vector<size_t> innerThreadCounts(outerThreadCount);
    if (::globParallelTree)
        // each inner thread count is approximately the total thread count divided by the outer thread count
        // and the sum of the inner thread counts is exactly the total thread count
        for (size_t outerThreadIndex = 0; outerThreadIndex < outerThreadCount; ++outerThreadIndex)
            innerThreadCounts[outerThreadIndex]
            = (totalThreadCount * (outerThreadIndex + 1) / outerThreadCount)
            - (totalThreadCount * outerThreadIndex / outerThreadCount);
    else
        std::fill(begin(innerThreadCounts), end(innerThreadCounts), 1);

    // since the profiling only tracks the master thread, we randomize
    // the initial item given to each thread and the inner thread counts
    std::shuffle(begin(optIndicesSortedByCost), begin(optIndicesSortedByCost) + outerThreadCount, theRne);
    std::shuffle(begin(innerThreadCounts), end(innerThreadCounts), theRne);

    // An OpenMP parallel for loop with dynamic scheduling does not in general process the elements in any 
    // particular order. Therefore we implement our own scheduling to make sure they are processed in order.

    std::atomic<size_t> nextSortedOptIndex = 0;
    BEGIN_EXCEPTION_SAFE_OMP_PARALLEL(static_cast<int>(outerThreadCount))
    {
        ASSERT(static_cast<size_t>(omp_get_num_threads()) == outerThreadCount);
        size_t outerThreadIndex = omp_get_thread_num();
        size_t innerThreadCount = innerThreadCounts[outerThreadIndex];

        while (true) {
            size_t sortedOptIndex = nextSortedOptIndex++;
            if (sortedOptIndex >= optCount) break;
            size_t optIndex = optIndicesSortedByCost[sortedOptIndex];
            pred[optIndex] = trainer.train(opt[optIndex], innerThreadCount);
        }
    }
    END_EXCEPTION_SAFE_OMP_PARALLEL(PROFILE::OUTER_THREAD_SYNCH);

    return pred;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXXdc parallelTrainAndPredict(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    CRefXXfc testInData)
{
    if (testInData.rows() == 0)
        throw std::invalid_argument("Test indata has 0 samples.");

    const size_t sampleCount = testInData.rows();
    const size_t optCount = size(opt);

    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(
        cbegin(opt),
        cend(opt),
        begin(optIndicesSortedByCost),
        [](const auto& opt) { return -opt.cost(); }
    );

    ArrayXXdc predData(sampleCount, optCount);

    const size_t totalThreadCount = omp_get_max_threads();

    size_t outerThreadCount;
    if (::globParallelTree) {
        omp_set_nested(true);
        outerThreadCount = ::globOuterThreadCount;
        if (outerThreadCount == 0)
            outerThreadCount = static_cast<size_t>(std::round(std::sqrt(totalThreadCount)));
        if (outerThreadCount > totalThreadCount)
            outerThreadCount = totalThreadCount;
    }
    else
        outerThreadCount = totalThreadCount;
    if (outerThreadCount > optCount)
        outerThreadCount = optCount;

    vector<size_t> innerThreadCounts(outerThreadCount);
    if (::globParallelTree)
        for (size_t outerThreadIndex = 0; outerThreadIndex < outerThreadCount; ++outerThreadIndex)
            innerThreadCounts[outerThreadIndex]
            = (totalThreadCount * (outerThreadIndex + 1) / outerThreadCount)
            - (totalThreadCount * outerThreadIndex / outerThreadCount);
    else
        std::fill(begin(innerThreadCounts), end(innerThreadCounts), 1);

    std::shuffle(begin(optIndicesSortedByCost), begin(optIndicesSortedByCost) + outerThreadCount, theRne);
    std::shuffle(begin(innerThreadCounts), end(innerThreadCounts), theRne);

    std::atomic<size_t> nextSortedOptIndex = 0;
    BEGIN_EXCEPTION_SAFE_OMP_PARALLEL(static_cast<int>(outerThreadCount))
    {
        ASSERT(static_cast<size_t>(omp_get_num_threads()) == outerThreadCount);
        size_t outerThreadIndex = omp_get_thread_num();
        size_t innerThreadCount = innerThreadCounts[outerThreadIndex];

        while (true) {
            size_t sortedOptIndex = nextSortedOptIndex++;
            if (sortedOptIndex >= optCount) break;
            size_t optIndex = optIndicesSortedByCost[sortedOptIndex];
            shared_ptr<BoostPredictor> pred = trainer.train(opt[optIndex], innerThreadCount);
            predData.col(optIndex) = pred->predict(testInData);
        }
    }
    END_EXCEPTION_SAFE_OMP_PARALLEL(PROFILE::OUTER_THREAD_SYNCH);

    return predData;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXd parallelTrainAndEval(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    CRefXXfc testInData, CRefXs testOutData, function<double(CRefXs, CRefXd)> lossFun
)
{
    size_t optCount = size(opt);

    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(
        cbegin(opt),
        cend(opt),
        begin(optIndicesSortedByCost),
        [](const auto& opt) { return -opt.cost(); }
    );

    ArrayXd scores(optCount);

    const size_t totalThreadCount = omp_get_max_threads();

    size_t outerThreadCount;
    if (::globParallelTree) {
        omp_set_nested(true);
        outerThreadCount = ::globOuterThreadCount;
        if (outerThreadCount == 0)
            outerThreadCount = static_cast<size_t>(std::round(std::sqrt(totalThreadCount)));
        if (outerThreadCount > totalThreadCount)
            outerThreadCount = totalThreadCount;
    }
    else
        outerThreadCount = totalThreadCount;
    if (outerThreadCount > optCount)
        outerThreadCount = optCount;

    vector<size_t> innerThreadCounts(outerThreadCount);
    if (::globParallelTree)
        for (size_t outerThreadIndex = 0; outerThreadIndex < outerThreadCount; ++outerThreadIndex)
            innerThreadCounts[outerThreadIndex]
            = (totalThreadCount * (outerThreadIndex + 1) / outerThreadCount)
            - (totalThreadCount * outerThreadIndex / outerThreadCount);
    else
        std::fill(begin(innerThreadCounts), end(innerThreadCounts), 1);

    std::shuffle(begin(optIndicesSortedByCost), begin(optIndicesSortedByCost) + outerThreadCount, theRne);
    std::shuffle(begin(innerThreadCounts), end(innerThreadCounts), theRne);

    //cout << "0";
    std::atomic<size_t> nextSortedOptIndex = 0;
    BEGIN_EXCEPTION_SAFE_OMP_PARALLEL(static_cast<int>(outerThreadCount))
    {
        ASSERT(static_cast<size_t>(omp_get_num_threads()) == outerThreadCount);
        size_t outerThreadIndex = omp_get_thread_num();
        size_t innerThreadCount = innerThreadCounts[outerThreadIndex];

        //stringstream ss;
        //ss << outerThreadCount << " -> " << innerThreadCount << '\n';
        //cout << ss.str();

        while (true) {
            size_t sortedOptIndex = nextSortedOptIndex++;
            if (sortedOptIndex >= optCount) break;
            size_t optIndex = optIndicesSortedByCost[sortedOptIndex];
            shared_ptr<BoostPredictor> pred = trainer.train(opt[optIndex], innerThreadCount);
            ArrayXd predData = pred->predict(testInData);
            scores(optIndex) = lossFun(testOutData, predData);
        }
        //cout << '.';
    }
    END_EXCEPTION_SAFE_OMP_PARALLEL(PROFILE::OUTER_THREAD_SYNCH);
    //cout << '\n';

    return scores;
}

ArrayXd parallelTrainAndEvalWeighted(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    CRefXXfc testInData, CRefXs testOutData, CRefXd testWeights, function<double(CRefXs, CRefXd, CRefXd)> lossFun
)
{
    size_t optCount = size(opt);

    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(
        cbegin(opt),
        cend(opt),
        begin(optIndicesSortedByCost),
        [](const auto& opt) { return -opt.cost(); }
    );

    ArrayXd scores(optCount);

    const size_t totalThreadCount = omp_get_max_threads();

    size_t outerThreadCount;
    if (::globParallelTree) {
        omp_set_nested(true);
        outerThreadCount = ::globOuterThreadCount;
        if (outerThreadCount == 0)
            outerThreadCount = static_cast<size_t>(std::round(std::sqrt(totalThreadCount)));
        if (outerThreadCount > totalThreadCount)
            outerThreadCount = totalThreadCount;
    }
    else
        outerThreadCount = totalThreadCount;
    if (outerThreadCount > optCount)
        outerThreadCount = optCount;

    vector<size_t> innerThreadCounts(outerThreadCount);
    if (::globParallelTree)
        for (size_t outerThreadIndex = 0; outerThreadIndex < outerThreadCount; ++outerThreadIndex)
            innerThreadCounts[outerThreadIndex]
            = (totalThreadCount * (outerThreadIndex + 1) / outerThreadCount)
            - (totalThreadCount * outerThreadIndex / outerThreadCount);
    else
        std::fill(begin(innerThreadCounts), end(innerThreadCounts), 1);

    std::shuffle(begin(optIndicesSortedByCost), begin(optIndicesSortedByCost) + outerThreadCount, theRne);
    std::shuffle(begin(innerThreadCounts), end(innerThreadCounts), theRne);

    std::atomic<size_t> nextSortedOptIndex = 0;
    BEGIN_EXCEPTION_SAFE_OMP_PARALLEL(static_cast<int>(outerThreadCount))
    {
        ASSERT(static_cast<size_t>(omp_get_num_threads()) == outerThreadCount);
        size_t outerThreadIndex = omp_get_thread_num();
        size_t innerThreadCount = innerThreadCounts[outerThreadIndex];

        while (true) {
            size_t sortedOptIndex = nextSortedOptIndex++;
            if (sortedOptIndex >= optCount) break;
            size_t optIndex = optIndicesSortedByCost[sortedOptIndex];
            shared_ptr<BoostPredictor> pred = trainer.train(opt[optIndex], innerThreadCount);
            ArrayXd predData = pred->predict(testInData);
            scores(optIndex) = lossFun(testOutData, predData, testWeights);
        }
    }
    END_EXCEPTION_SAFE_OMP_PARALLEL(PROFILE::OUTER_THREAD_SYNCH);

    return scores;
}

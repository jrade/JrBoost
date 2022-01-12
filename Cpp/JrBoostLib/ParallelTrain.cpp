//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "ParallelTrain.h"

#include "BoostOptions.h"
#include "BoostTrainer.h"
#include "OmpParallel.h"
#include "Predictor.h"


static double cost_(const BoostOptions& opt)
{
    return opt.iterationCount() * opt.forestSize() * opt.maxTreeDepth() * opt.usedSampleRatio() * opt.topVariableCount()
           * opt.usedVariableRatio();
}

static size_t outerThreadCount_(size_t threadCount)
{
    if (threadCount <= 8)
        return threadCount;
    return static_cast<size_t>(std::round(std::sqrt(8.0 * threadCount)));
}

//----------------------------------------------------------------------------------------------------------------------

vector<shared_ptr<Predictor>> parallelTrain(const BoostTrainer& trainer, const vector<BoostOptions>& opt)
{
    vector<shared_ptr<Predictor>> pred;
    GUARDED_PROFILE_PUSH(PROFILE::OUTER_THREAD_SYNCH);

    const size_t optCount = size(opt);
    pred.resize(optCount);

    // In order to keep the threads balanced we will process the options one-by-one in order
    // from the most computaionally expensive to the least computationally expensive
    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(cbegin(opt), cend(opt), begin(optIndicesSortedByCost), [](const auto& opt) { return -cost_(opt); });

    const size_t threadCount = omp_get_max_threads();
    const size_t outerThreadCount = std::min(optCount, outerThreadCount_(threadCount));

    // An OpenMP parallel for loop with dynamic scheduling does not in general process the elements in any
    // particular order. Therefore we implement our own scheduling to make sure they are processed in order.
    std::atomic<size_t> nextSortedOptIndex = 0;
    BEGIN_OMP_PARALLEL(outerThreadCount)
    {
        const size_t outerThreadIndex = omp_get_thread_num();
        const size_t innerThreadCount = (threadCount * (outerThreadIndex + 1)) / outerThreadCount
                                        - (threadCount * outerThreadIndex) / outerThreadCount;
        while (true) {
            const size_t sortedOptIndex = nextSortedOptIndex++;
            if (sortedOptIndex >= optCount)
                break;
            size_t optIndex = optIndicesSortedByCost[sortedOptIndex];
            pred[optIndex] = trainer.train(opt[optIndex], innerThreadCount);
        }
    }
    END_OMP_PARALLEL

    GUARDED_PROFILE_POP;
    return pred;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXXdc parallelTrainAndPredict(const BoostTrainer& trainer, const vector<BoostOptions>& opt, CRefXXfc testInData)
{
    ArrayXXdc predData;
    GUARDED_PROFILE_PUSH(PROFILE::OUTER_THREAD_SYNCH);

    if (testInData.rows() == 0)
        throw std::invalid_argument("Test indata has 0 samples.");

    const size_t sampleCount = testInData.rows();
    const size_t optCount = size(opt);
    predData = ArrayXXdc(sampleCount, optCount);

    // In order to keep the threads balanced we will process the options one-by-one in order
    // from the most computaionally expensive to the least computationally expensive
    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(cbegin(opt), cend(opt), begin(optIndicesSortedByCost), [](const auto& opt) { return -cost_(opt); });

    const size_t threadCount = omp_get_max_threads();
    const size_t outerThreadCount = std::min(optCount, outerThreadCount_(threadCount));

    // An OpenMP parallel for loop with dynamic scheduling does not in general process the elements in any
    // particular order. Therefore we implement our own scheduling to make sure they are processed in order.
    std::atomic<size_t> nextSortedOptIndex = 0;
    BEGIN_OMP_PARALLEL(outerThreadCount)
    {
        const size_t outerThreadIndex = omp_get_thread_num();
        const size_t innerThreadCount = (threadCount * (outerThreadIndex + 1)) / outerThreadCount
                                        - (threadCount * outerThreadIndex) / outerThreadCount;
        while (true) {
            const size_t sortedOptIndex = nextSortedOptIndex++;
            if (sortedOptIndex >= optCount)
                break;
            size_t optIndex = optIndicesSortedByCost[sortedOptIndex];
            shared_ptr<Predictor> pred = trainer.train(opt[optIndex], innerThreadCount);
            predData.col(optIndex) = pred->predict(testInData, innerThreadCount);
        }
    }
    END_OMP_PARALLEL

    GUARDED_PROFILE_POP;
    return predData;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXd parallelTrainAndEval(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    function<double(CRefXu8, CRefXd, optional<CRefXd>)> lossFun, CRefXXfc testInData, CRefXu8 testOutData,
    optional<CRefXd> testWeights)
{
    ArrayXd scores;
    GUARDED_PROFILE_PUSH(PROFILE::OUTER_THREAD_SYNCH);

    const size_t optCount = size(opt);
    scores = ArrayXd(optCount);

    // In order to keep the threads balanced we will process the options one-by-one in order
    // from the most computaionally expensive to the least computationally expensive
    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(cbegin(opt), cend(opt), begin(optIndicesSortedByCost), [](const auto& opt) { return -cost_(opt); });

    const size_t threadCount = omp_get_max_threads();
    const size_t outerThreadCount = std::min(optCount, outerThreadCount_(threadCount));

    // An OpenMP parallel for loop with dynamic scheduling does not in general process the elements in any
    // particular order. Therefore we implement our own scheduling to make sure they are processed in order.
    std::atomic<size_t> nextSortedOptIndex = 0;
    BEGIN_OMP_PARALLEL(outerThreadCount)
    {
        const size_t outerThreadIndex = omp_get_thread_num();
        const size_t innerThreadCount = (threadCount * (outerThreadIndex + 1)) / outerThreadCount
                                        - (threadCount * outerThreadIndex) / outerThreadCount;
        while (true) {
            const size_t sortedOptIndex = nextSortedOptIndex++;
            if (sortedOptIndex >= optCount)
                break;
            size_t optIndex = optIndicesSortedByCost[sortedOptIndex];
            shared_ptr<Predictor> pred = trainer.train(opt[optIndex], innerThreadCount);
            ArrayXd predData = pred->predict(testInData, innerThreadCount);
            scores(optIndex) = lossFun(testOutData, predData, testWeights);
        }
    }
    END_OMP_PARALLEL

    GUARDED_PROFILE_POP;
    return scores;
}

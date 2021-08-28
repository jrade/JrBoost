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


// In each iteration of the loop we build a classifier with one set of options
// Differents sets of options can take very different time.
// To ensure that the OMP threads are balanced we sort the options objects
// from the most time-consuming to the least.
// Note that we can not use 
//      #pragma omp for schedule(dynamic)
// because it is not guarangeed to process the elements in order

//----------------------------------------------------------------------------------------------------------------------

vector<shared_ptr<BoostPredictor>> parallelTrain(const BoostTrainer& trainer, const vector<BoostOptions>& opt)
{
    const size_t optCount = size(opt);

    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(
        cbegin(opt),
        cend(opt),
        begin(optIndicesSortedByCost),
        [](const auto& opt) { return -opt.cost(); }
    );

    vector<shared_ptr<BoostPredictor>> pred(optCount);
    std::atomic<int> i0 = 0;

    BEGIN_EXCEPTION_SAFE_OMP_PARALLEL
    {
        while (true) {
            size_t i = i0++;
            if (i >= size(opt)) break;
            size_t j = optIndicesSortedByCost[i];
            pred[j] = trainer.train(opt[j]);
        }
    }
    END_EXCEPTION_SAFE_OMP_PARALLEL(PROFILE::THREAD_SYNCH);

    return pred;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXXd parallelTrainAndPredict(const BoostTrainer& trainer, const vector<BoostOptions>& opt, CRefXXf testInData)
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

    ArrayXXd predData(sampleCount, optCount);
    std::atomic<int> i0 = 0;

    BEGIN_EXCEPTION_SAFE_OMP_PARALLEL
    {
        while (true) {
            size_t i = i0++;
            if (i >= size(opt)) break;
            size_t j = optIndicesSortedByCost[i];
            shared_ptr<BoostPredictor> pred = trainer.train(opt[j]);
            predData.col(j) = pred->predict(testInData);
        }
    }
    END_EXCEPTION_SAFE_OMP_PARALLEL(PROFILE::THREAD_SYNCH);

    return predData;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXd parallelTrainAndEval(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    CRefXXf testInData, CRefXs testOutData, function<double(CRefXs, CRefXd)> lossFun
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
    std::atomic<int> i0 = 0;

    BEGIN_EXCEPTION_SAFE_OMP_PARALLEL
    {
        while (true) {
            size_t i = i0++;
            if (i >= size(opt)) break;
            size_t j = optIndicesSortedByCost[i];
            shared_ptr<BoostPredictor> pred = trainer.train(opt[j]);
            ArrayXd predData = pred->predict(testInData);
            scores(j) = lossFun(testOutData, predData);
        }
    }
    END_EXCEPTION_SAFE_OMP_PARALLEL(PROFILE::THREAD_SYNCH);

    return scores;
}

ArrayXd parallelTrainAndEvalWeighted(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    CRefXXf testInData, CRefXs testOutData, CRefXd testWeights, function<double(CRefXs, CRefXd, CRefXd)> lossFun
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
    std::atomic<int> i0 = 0;

    BEGIN_EXCEPTION_SAFE_OMP_PARALLEL
    {
        while (true) {
            size_t i = i0++;
            if (i >= size(opt)) break;
            size_t j = optIndicesSortedByCost[i];
            shared_ptr<BoostPredictor> pred = trainer.train(opt[j]);
            ArrayXd predData = pred->predict(testInData);
            scores(j) = lossFun(testOutData, predData, testWeights);
        }
    }
    END_EXCEPTION_SAFE_OMP_PARALLEL(PROFILE::THREAD_SYNCH);

    return scores;
}

#include "pch.h"
#include "StumpTrainer.h"
#include "StumpTrainerShared.h"
#include "StumpTrainerByThread.h"
#include "StumpPredictor.h"


StumpTrainer::StumpTrainer(CRefXXf inData, RefXs strata) :
    sampleCount_(inData.rows())
{
    ASSERT(inData.rows() != 0);
    ASSERT(inData.cols() != 0);
    ASSERT(static_cast<size_t>(strata.rows()) == sampleCount_);

    ASSERT((inData > -numeric_limits<float>::infinity()).all());
    ASSERT((inData < numeric_limits<float>::infinity()).all());
    ASSERT((strata < 2).all());

    shared_ = std::make_shared<const StumpTrainerShared>(inData, strata);
    std::random_device rd;
    size_t threadCount = omp_get_max_threads();
    for (size_t threadId = 0; threadId < threadCount; ++threadId)
        byThread_.push_back(std::make_shared<StumpTrainerByThread>(inData, shared_, rd));
}


StumpPredictor StumpTrainer::train(CRefXd outData, CRefXd weights, const StumpOptions& options) const
{
    ASSERT(static_cast<size_t>(outData.rows()) == sampleCount_);
    ASSERT((outData > -numeric_limits<double>::infinity()).all());
    ASSERT((outData < numeric_limits<double>::infinity()).all());

    ASSERT(static_cast<size_t>(weights.rows()) == sampleCount_);
    ASSERT((weights >= 0.0).all());
    ASSERT((weights < numeric_limits<double>::infinity()).all());

    int threadId = omp_get_thread_num();
    return byThread_[threadId]->train(outData, weights, options);
}

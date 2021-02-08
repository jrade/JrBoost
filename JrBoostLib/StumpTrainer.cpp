#include "pch.h"
#include "StumpTrainer.h"
#include "StumpTrainerShared.h"
#include "StumpTrainerByThread.h"
#include "AbstractPredictor.h"


StumpTrainer::StumpTrainer(CRefXXf inData, RefXs strata) :
    sampleCount_{ static_cast<size_t>(inData.rows()) },
    shared_{ std::make_shared<const StumpTrainerShared>(inData, strata) }
{
    ASSERT(inData.rows() != 0);
    ASSERT(inData.cols() != 0);
    ASSERT(static_cast<size_t>(strata.rows()) == sampleCount_);

    ASSERT((inData > -numeric_limits<float>::infinity()).all());
    ASSERT((inData < numeric_limits<float>::infinity()).all());
    ASSERT((strata < 2).all());

    std::random_device rd;
    size_t threadCount = omp_get_max_threads();
    for (int threadId = 0; threadId < threadCount; ++threadId)
        byThread_.push_back(std::make_shared<StumpTrainerByThread>(inData, shared_, rd));
}


unique_ptr<AbstractPredictor> StumpTrainer::train(CRefXd outData, CRefXd weights, const StumpOptions& options) const
{
    PROFILE::PUSH(PROFILE::ST_TRAIN);
    PROFILE::PUSH(PROFILE::ST_VAL);

    ASSERT(static_cast<size_t>(outData.rows()) == sampleCount_);
    ASSERT((outData > -numeric_limits<double>::infinity()).all());
    ASSERT((outData < numeric_limits<double>::infinity()).all());

    ASSERT(static_cast<size_t>(weights.rows()) == sampleCount_);
    ASSERT((weights >= 0.0).all());
    ASSERT((weights < numeric_limits<double>::infinity()).all());

    PROFILE::POP(sampleCount_);

    int threadId = omp_get_thread_num();
    unique_ptr<AbstractPredictor> pred = byThread_[threadId]->train(outData, weights, options);

    PROFILE::POP();

    return pred;
}

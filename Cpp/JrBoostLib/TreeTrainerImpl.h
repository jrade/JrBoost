//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BernoulliDistribution.h"
#include "TreeNodeTrainer.h"
#include "TreeNodeExt.h"
#include "TreeTrainerImplBase.h"

class TreeOptions;
class BasePredictor;

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
class TreeTrainerImpl : public TreeTrainerImplBase 
{
public:
    TreeTrainerImpl(CRefXXf inData, CRefXs strata);
    virtual ~TreeTrainerImpl() = default;

    virtual unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const TreeOptions& options, size_t threadCount) const;

private:
    using BernoulliDistribution = typename std::conditional<
        sizeof(SampleIndex) == 8,
        FastBernoulliDistribution,
        VeryFastBernoulliDistribution
    >::type;

private:
    vector<vector<SampleIndex>> initSortedSamples_() const;

    void validateData_(CRefXd outData, CRefXd weights) const;
    pair<size_t, size_t> initUsedVariables_(const TreeOptions& opionst) const;
    size_t initSampleStatus_(CRefXd weights, const TreeOptions& options) const;
    void initRoot_(CRefXd outData, CRefXd weights, size_t usedSampleCount) const;

    const SampleIndex* initOrderedSamplesFast_(size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& opions, size_t d) const;
    const SampleIndex* initOrderedSamples_(size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& opions, size_t d) const;
    const SampleIndex* updateOrderedSamples_(size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& opions, size_t d) const;

    void initSplits_(const TreeOptions& options, size_t d, size_t threadCount) const;
    void updateSplits_(CRefXd outData, CRefXd weights, const TreeOptions& options,
        const SampleIndex* orderedSamples, size_t usedVariableIndex, size_t d, size_t threadIndex) const;
    void joinSplits_(size_t d, size_t threadCount) const;
    size_t finalizeSplits_(size_t d) const;

    size_t updateSampleStatus_(CRefXd outData, CRefXd weights, size_t d) const;
    unique_ptr<BasePredictor> initPredictor_() const;

private:
    const CRefXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const vector<vector<SampleIndex>> sortedSamples_;
    const CRefXs strata_;
    const size_t stratum0Count_;
    const size_t stratum1Count_;

private:
    struct OuterThreadData_
    {
        vector<size_t> usedVariables;
        vector<vector<TreeNodeExt>> tree;
        vector<TreeNodeTrainer<SampleIndex>> treeNodeTrainers;
        vector<SampleIndex> sampleStatus;
        vector<vector<SampleIndex>> sampleBufferByVariable;
    };

    inline static thread_local OuterThreadData_ out_;

    struct InnerThreadData_
    {
        OuterThreadData_* out = nullptr;    // pointer to thread local data of parent thread
        vector<SampleIndex> sampleBuffer;
        vector<SampleIndex*> samplePointerBuffer;
    };
    
    inline static thread_local InnerThreadData_ in_;
};

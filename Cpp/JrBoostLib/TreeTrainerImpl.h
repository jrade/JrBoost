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
    TreeTrainerImpl(CRefXXfc inData, CRefXs strata);
    virtual ~TreeTrainerImpl() = default;

    virtual unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const TreeOptions& options, size_t threadCount) const;

private:
    struct ThreadLocalData_;

private:
    vector<vector<SampleIndex>> createSortedSamples_() const;

    void validateData_(CRefXd outData, CRefXd weights) const;
    size_t initUsedVariables_(const TreeOptions& opionst) const;
    size_t initSampleStatus_(CRefXd weights, const TreeOptions& options) const;
    void initRoot_(CRefXd outData, CRefXd weights, size_t usedSampleCount) const;

    const SampleIndex* initOrderedSamplesFast_(
        size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& opions, size_t d) const;
    const SampleIndex* initOrderedSamples_(
        size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& opions, size_t d) const;
    const SampleIndex* updateOrderedSamples_(
        size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& opions, size_t d) const;

    void initNodeTrainers_(const TreeOptions& options, size_t d, size_t threadCount) const;
    void updateNodeTrainers_(CRefXd outData, CRefXd weights, const TreeOptions& options,
        const SampleIndex* orderedSamples, size_t usedVariableIndex, size_t d) const;
    void joinNodeTrainers_(size_t d, size_t threadCount) const;
    bool finalizeNodeTrainers_(size_t d) const;

    size_t updateSampleStatus_(CRefXd outData, CRefXd weights, size_t d) const;
    unique_ptr<BasePredictor> createPredictor_() const;

private:
    const CRefXXfc inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const vector<vector<SampleIndex>> sortedSamples_;

    const CRefXs strata_;
    const size_t stratum0Count_;
    const size_t stratum1Count_;

    inline static thread_local ThreadLocalData_ threadLocalData_;

private:
    struct ThreadLocalData_
    {
        ThreadLocalData_* parent = nullptr;
        // threadlocal data of parent thread

        vector<size_t> usedVariables;
        vector<vector<TreeNodeExt>> tree;
        vector<CacheLineAligned<TreeNodeTrainer<SampleIndex>>> treeNodeTrainers;

        vector<SampleIndex> sampleStatus;
        // status of each sample in the current layer of the tree
        // status`= 0 means the sample is unused
        // status = k + 1 means the sample belongs to node number k in the layer; k = 0, 1, ..., node count - 1

        vector<vector<SampleIndex>> orderedSamplesByVariable;
        // orderedSamplesByVariable[j] contains the active samples grouped by node
        // and then sorted by the j-th used variable
        // (only used if options.saveMemory() = false)

        vector<SampleIndex> sampleBuffer;
        vector<SampleIndex*> samplePointerBuffer;
        // tmp buffers
    };

    using BernoulliDistribution_ = typename std::conditional<
        sizeof(SampleIndex) == 8,
        FastBernoulliDistribution,
        VeryFastBernoulliDistribution
    >::type;
    // much faster than std::bernoulli_distribution
};

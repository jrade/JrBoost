//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "AGRandom.h"
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

    virtual unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const TreeOptions& options) const;

private:
    using RandomNumberEngine_ = splitmix;

    struct Split
    {
        double sumW;
        double sumWY;
        double minNodeWeight;

        bool splitFound;
        double score;
        size_t j;
        float x;
        double leftY;
        double rightY;

        size_t iterationCount;
        size_t slowBranchCount;
    };

    using BernoulliDistribution = typename std::conditional<
        sizeof(SampleIndex) == 8,
        FastBernoulliDistribution,
        VeryFastBernoulliDistribution
    >::type;

    vector<vector<SampleIndex>> initSortedSamples_() const;

    void validateData_(CRefXd outData, CRefXd weights) const;
    pair<size_t, size_t> initUsedVariables_(const TreeOptions& opionst) const;
    size_t initSampleStatus_(CRefXd weights, const TreeOptions& options) const;
    void initRoot_(CRefXd outData, CRefXd weights, size_t usedSampleCount) const;

    const SampleIndex* initOrderedSamplesFast_(size_t usedVariableIndex, size_t usedSampleCount, size_t d) const;
    const SampleIndex* initOrderedSamples_(size_t usedVariableIndex, size_t usedSampleCount, size_t d) const;
    const SampleIndex* initOrderedSamplesAlt_(size_t usedVariableIndex, size_t usedSampleCount, size_t d) const;
    const SampleIndex* updateOrderedSamplesAlt_(size_t usedVariableIndex, size_t usedSampleCount, size_t d) const;
    //void initOrderedSamplesByNode_(span<SampleIndex> orderedSamples, size_t d) const;

    void initSplits_(const TreeOptions& options, size_t d) const;
    void updateSplits_(CRefXd outData, CRefXd weights, const TreeOptions& options,
        const SampleIndex* orderedSamples, size_t usedVariableIndex, size_t d) const;
    size_t finalizeSplits_(size_t d) const;

    size_t updateSampleStatus_(CRefXd outData, CRefXd weights, size_t d) const;
    unique_ptr<BasePredictor> initPredictor_() const;

    const CRefXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const vector<vector<SampleIndex>> sortedSamples_;
    const CRefXs strata_;
    const size_t stratum0Count_;
    const size_t stratum1Count_;

    inline static thread_local RandomNumberEngine_ tlRne_;
    inline static thread_local vector<size_t> tlUsedVariables_;
    inline static thread_local vector<vector<TreeNodeExt>> tlTree_;
    inline static thread_local vector<TreeNodeTrainer<SampleIndex>> tlTreeNodeTrainers_;

    inline static thread_local vector<SampleIndex> tlSampleStatus_;
    inline static thread_local vector<SampleIndex> tlSampleBuffer_;
    inline static thread_local vector<vector<SampleIndex>> tlSampleBufferByVariable_;
    inline static thread_local vector<SampleIndex*> tlSamplePointerBuffer_;

    inline static thread_local struct ThreadLocalInit_ {
        ThreadLocalInit_() {
            std::random_device rd;
            tlRne_.seed(rd);
        }
    } threadLocalInit_{};
};

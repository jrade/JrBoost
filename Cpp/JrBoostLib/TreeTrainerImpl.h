// Copyright (C) 2021 Johan Rade <johan.rade@gmail.com>
//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "AGRandom.h"
#include "BernoulliDistribution.h"
#include "TreeNode.h"
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

    struct Split {
        bool isInit;
        double sumW;
        double sumWY;
        double minNodeWeight;

        bool splitFound;
        double score;
        size_t j;
        float x;
        double leftY;
        double rightY;
        size_t leftSampleCount;
        size_t rightSampleCount;

        size_t iterationCount;
        size_t slowBranchCount;
    };

    using BernoulliDistribution = typename std::conditional<
        sizeof(SampleIndex) == 8,
        FastBernoulliDistribution,
        VeryFastBernoulliDistribution
    >::type;

    vector<vector<SampleIndex>> createSortedSamples_() const;

    void validateData_(CRefXd outData, CRefXd weights) const;
    pair<size_t, size_t> initUsedVariables_(const TreeOptions& opt) const;
    size_t initSampleStatus_(const TreeOptions& opt, CRefXd weights) const;
    void initTree_(size_t usedSampleCount, const TreeOptions& options) const;
    void initSplits_(size_t d) const;
    void initOrderedSamples_(size_t usedVariableIndex, size_t usedSampleCount) const;
    void initOrderedSamplesAlt_(size_t usedVariableIndex, size_t usedSampleCount) const;
    void updateOrderedSamples_(size_t usedVariableIndex, size_t usedSampleCount) const;
    void updateOrderedSamplesAlt_(size_t usedVariableIndex, size_t usedSampleCount, size_t d) const;
    void initOrderedSampleIndex_(std::span<SampleIndex> orderedSamples) const;
    void updateSplits_(size_t j, CRefXd outData, CRefXd weights, const TreeOptions& options) const;
    void updateSplit_(
        Split* split, size_t j, std::span<SampleIndex> usedSamples,
        CRefXd outData, CRefXd weights, const TreeOptions& options) const;
    size_t updateTree_(size_t d) const;
    void updateSampleStatus_(size_t d) const;
    unique_ptr<BasePredictor> createPredictor_() const;

    const CRefXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const vector<vector<SampleIndex>> sortedSamples_;
    const CRefXs strata_;
    const size_t stratum0Count_;
    const size_t stratum1Count_;

    // outer data only used by outer threads [in yet to be implemented version]
    inline static thread_local RandomNumberEngine_ tlRne_;

    // outer data also used by inner threads (beware!)
    inline static thread_local vector<size_t> tlUsedVariables_;
    inline static thread_local vector<vector<TreeNode>> tlTree_;
    inline static thread_local vector<size_t> tlSampleCountByParentNode_;
    inline static thread_local vector<size_t> tlSampleCountByChildNode_;
    inline static thread_local vector<SampleIndex> tlSampleStatus_;
    inline static thread_local vector<vector<SampleIndex>> tlOrderedSamplesByVariable_;

    // need both outer and inner version
    inline static thread_local vector<Split> tlSplits_;

    // inner data (only used by inner threads)
    inline static thread_local vector<SampleIndex> tlSampleBuffer_;
    inline static thread_local vector<typename std::span<SampleIndex>::iterator> tlOrderedSampleIndex_;

    inline static thread_local struct ThreadLocalInit_ {
        ThreadLocalInit_() {
            std::random_device rd;
            tlRne_.seed(rd);
        }
    } threadLocalInit_{};
};

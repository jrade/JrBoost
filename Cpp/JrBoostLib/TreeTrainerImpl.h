//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BernoulliDistribution.h"
#include "TreeTrainer.h"
#include "TreeTrainerBuffers.h"


template<typename SampleIndex>
class TreeTrainerImpl : public TreeTrainer, private TreeTrainerBuffers {   // immutable class
public:
    TreeTrainerImpl(CRefXXfc inData, CRefXu8 strata);
    virtual ~TreeTrainerImpl() = default;

private:
    vector<size_t> initSampleCountByStratum_() const;
    vector<vector<SampleIndex>> initSortedSamples_() const;

    unique_ptr<BasePredictor>
    trainImpl0_(CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const;

    template<typename SampleStatus>
    void trainImpl1_(CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const;

    void validateData_(CRefXd outData, CRefXd weights) const;
#if PACKED_DATA
    void initWyPacks(CRefXd outData, CRefXd weights) const;
#endif
    size_t usedVariableCount_(const BaseOptions& options) const;
    size_t initUsedVariables_(const BaseOptions& options) const;
    void initTree_() const;

    template<typename SampleStatus>
    size_t initSampleStatus_(CRefXd outData, CRefXd weights, const BaseOptions& options) const;
    template<typename SampleStatus>
    void updateSampleStatus_(CRefXd outData, CRefXd weights, size_t d) const;

    template<typename SampleStatus>
    const SampleIndex*
    initOrderedSamples_(size_t usedVariableIndex, size_t usedSampleCount, const BaseOptions& opions, size_t d) const;
    template<typename SampleStatus>
    const SampleIndex* updateOrderedSampleSaveMemory_(
        size_t usedVariableIndex, size_t usedSampleCount, const BaseOptions& opions, size_t d) const;
    template<typename SampleStatus>
    const SampleIndex*
    updateOrderedSamples_(size_t usedVariableIndex, size_t usedSampleCount, const BaseOptions& opions, size_t d) const;

    void initNodeTrainers_(const BaseOptions& options, size_t d, size_t threadCount) const;
    void updateNodeTrainers_(
#if !PACKED_DATA
        CRefXd outData, CRefXd weights,
#endif
        const SampleIndex* orderedSamples, size_t usedVariableIndex, size_t d) const;
    size_t finalizeNodeTrainers_(size_t d, size_t threadCount) const;

    unique_ptr<BasePredictor> createPredictor_() const;

private:
    const CRefXXfc inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const vector<vector<SampleIndex>> sortedSamples_;

    const CRefXu8 strata_;
    const size_t stratumCount_;
    const vector<size_t> sampleCountByStratum_;

private:
    using BernoulliDistribution_ = typename std::conditional_t<   // much faster than std::bernoulli_distribution
        sizeof(SampleIndex) == 8, FastBernoulliDistribution, VeryFastBernoulliDistribution>;

    using Buffers = TreeTrainerBuffers;
};

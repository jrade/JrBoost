//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
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
    struct TrainData_ {
        CRefXd outData;
        CRefXd weights;
        BaseOptions options;
        size_t usedVariableCount;
        size_t threadCount;
    };

private:
    vector<size_t> initSampleCountsByStratum() const;

    vector<vector<SampleIndex>> initSortedSamples_() const;

    //

    unique_ptr<BasePredictor>
    trainImpl0_(CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const;

    void validateData_(CRefXd outData, CRefXd weights) const;

#if USE_PACKED_DATA
    void initWyPacks_(CRefXd outData, CRefXd weights) const;
#endif

    void initTree_() const;

    size_t usedVariableCount_(const BaseOptions& options) const;

    //

    template<typename SampleStatus>
    size_t trainImpl1_(const TrainData_* trainData, size_t ITEM_COUNT) const;

    template<typename SampleStatus>
    size_t initSampleStatus_(const TrainData_* trainData) const;

    template<typename SampleStatus>
    void updateSampleStatus_(const TrainData_* trainData, size_t d) const;

    size_t initUsedVariables_(const TrainData_* trainData) const;

    void initNodeTrainers_(const TrainData_* trainData, size_t d) const;

    size_t finalizeNodeTrainers_(const TrainData_* trainData, size_t d) const;

    //

    template<typename SampleStatus>
    size_t trainImpl2_(
        const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t usedVariableIndex, size_t threadIndex,
        size_t ITEM_COUNT) const;

    template<typename SampleStatus>
    const SampleIndex*
    initOrderedSamples_(const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t usedVariableIndex) const;

    template<typename SampleStatus>
    const SampleIndex* updateOrderedSampleSaveMemory_(
        const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t usedVariableIndex) const;

    template<typename SampleStatus>
    const SampleIndex* updateOrderedSamples_(
        const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t usedVariableIndex) const;

    void updateNodeTrainers_(
        const TrainData_* trainData, size_t d, const SampleIndex* orderedSamples, size_t usedVariableIndex,
        size_t threadIndex) const;

private:
    const CRefXXfc inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const vector<vector<SampleIndex>> sortedSamples_;

    const CRefXu8 strata_;
    const size_t stratumCount_;
    const vector<size_t> sampleCountsByStratum_;

private:
    using BernoulliDistribution_ = typename std::conditional_t<   // much faster than std::bernoulli_distribution
        sizeof(SampleIndex) == 8, FastBernoulliDistribution, VeryFastBernoulliDistribution>;
};

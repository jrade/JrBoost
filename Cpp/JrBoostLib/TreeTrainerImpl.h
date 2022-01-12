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
    struct TmpData_;

    vector<size_t> initSampleCountByStratum_() const;
    vector<vector<SampleIndex>> initSortedSamples_() const;

    unique_ptr<BasePredictor>
    trainImpl0_(CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const;

    template<typename SampleStatus>
    size_t trainImpl1_(TmpData_* tmpData, size_t ITEM_COUNT) const;

    template<typename SampleStatus>
    size_t trainImpl2_(TmpData_* tmpData, size_t usedVariableIndex, size_t threadIndex, size_t ITEM_COUNT) const;

    void validateData_(TmpData_* tmpData) const;
#if USE_PACKED_DATA
    void initWyPacks_(TmpData_* tmpData) const;
#endif
    size_t usedVariableCount_(TmpData_* tmpData) const;
    size_t initUsedVariables_(TmpData_* tmpData) const;
    void initTree_() const;

    template<typename SampleStatus>
    void initSampleStatus_(TmpData_* tmpData) const;
    template<typename SampleStatus>
    void updateSampleStatus_(TmpData_* tmpData) const;

    template<typename SampleStatus>
    const SampleIndex* initOrderedSamples_(TmpData_* tmpData, size_t usedVariableIndex) const;
    template<typename SampleStatus>
    const SampleIndex* updateOrderedSampleSaveMemory_(TmpData_* tmpData, size_t usedVariableIndex) const;
    template<typename SampleStatus>
    const SampleIndex* updateOrderedSamples_(TmpData_* tmpData, size_t usedVariableIndex) const;

    void initNodeTrainers_(TmpData_* tmpData) const;
    void updateNodeTrainers_(
        TmpData_* tmpData, const SampleIndex* orderedSamples, size_t usedVariableIndex, size_t threadIndex) const;
    void finalizeNodeTrainers_(TmpData_* tmpData) const;

private:
    struct TmpData_ {
        CRefXd outData;
        CRefXd weights;
        BaseOptions options;
        size_t threadCount;
        size_t d;
        size_t usedSampleCount;
    };

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

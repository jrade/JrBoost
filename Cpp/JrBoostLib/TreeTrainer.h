//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BernoulliDistribution.h"
#include "TreeTrainerBase.h"

class BaseOptions;
class BasePredictor;


template<typename SampleIndex>
class TreeTrainer : public TreeTrainerBase
{
public:
    TreeTrainer(CRefXXfc inData, CRefXu8 strata);
    virtual ~TreeTrainer() = default;

    virtual unique_ptr<BasePredictor> train(
        CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const;
private:
    vector<vector<SampleIndex>> getSortedSamples_() const;

    template<typename SampleStatus> unique_ptr<BasePredictor> trainImpl_(
        CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const;

    //void validateData_(CRefXd outData, CRefXd weights) const;
#if PACKED_DATA
    void initWyPacks(CRefXd outData, CRefXd weights) const;
#endif
    size_t usedVariableCount_(const BaseOptions& options) const;
    size_t initUsedVariables_(const BaseOptions& options) const;
    void initTree_() const;

    template<typename SampleStatus> size_t initSampleStatus_(CRefXd outData, CRefXd weights, const BaseOptions& options) const;
    template<typename SampleStatus> void updateSampleStatus_(CRefXd outData, CRefXd weights, size_t d) const;

    template<typename SampleStatus> const SampleIndex* initOrderedSamples_(
        size_t usedVariableIndex, size_t usedSampleCount, const BaseOptions& opions, size_t d) const;
    template<typename SampleStatus> const SampleIndex* updateOrderedSampleSaveMemory_(
        size_t usedVariableIndex, size_t usedSampleCount, const BaseOptions& opions, size_t d) const;
    template<typename SampleStatus> const SampleIndex* updateOrderedSamples_(
        size_t usedVariableIndex, size_t usedSampleCount, const BaseOptions& opions, size_t d) const;

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
    const size_t stratum0Count_;
    const size_t stratum1Count_;

private:
    using BernoulliDistribution_ = typename std::conditional<
        sizeof(SampleIndex) == 8,
        FastBernoulliDistribution,
        VeryFastBernoulliDistribution
    >::type;
    // much faster than std::bernoulli_distribution
};

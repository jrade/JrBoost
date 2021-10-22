//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BernoulliDistribution.h"
#include "TreeNodeTrainer.h"
#include "TreeTrainerImplBase.h"

class TreeOptions;
class BasePredictor;

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
class TreeTrainerImpl : public TreeTrainerImplBase
{
public:
    TreeTrainerImpl(CRefXXfc inData, CRefXu8 strata);
    virtual ~TreeTrainerImpl() = default;

    virtual unique_ptr<BasePredictor> train(
        CRefXd outData, CRefXd weights, const TreeOptions& options, size_t threadCount) const;
private:
    vector<vector<SampleIndex>> getSortedSamples_() const;

    template<typename SampleStatus> unique_ptr<BasePredictor> trainImpl_(
        CRefXd outData, CRefXd weights, const TreeOptions& options, size_t threadCount) const;

    void validateData_(CRefXd outData, CRefXd weights) const;
    size_t usedVariableCount_(const TreeOptions& options) const;
    size_t initUsedVariables_(const TreeOptions& options) const;

    template<typename SampleStatus> size_t initRoot_(CRefXd outData, CRefXd weights) const;

    template<typename SampleStatus> void initSampleStatus_(CRefXd weights, const TreeOptions& options) const;
    template<typename SampleStatus> void updateSampleStatus_(CRefXd outData, CRefXd weights, size_t d) const;

    template<typename SampleStatus> const SampleIndex* initOrderedSamplesLayer0_(
        size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& opions, size_t d) const;
    template<typename SampleStatus> const SampleIndex* initOrderedSamples_(
        size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& opions, size_t d) const;
    template<typename SampleStatus> const SampleIndex* updateOrderedSamples_(
        size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& opions, size_t d) const;

    void initNodeTrainers_(const TreeOptions& options, size_t d, size_t threadCount) const;
    void updateNodeTrainers_(CRefXd outData, CRefXd weights,
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

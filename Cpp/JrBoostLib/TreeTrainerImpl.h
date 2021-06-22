// Copyright (C) 2021 Johan Rade <johan.rade@gmail.com>
//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "AGRandom.h"
#include "BernoulliDistribution.h"

class StumpOptions;
class BasePredictor;


class TreeTrainerImplBase {
public:
    virtual ~TreeTrainerImplBase() = default;
    virtual unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const = 0;

protected:
    TreeTrainerImplBase() = default;

// deleted:
    TreeTrainerImplBase(const TreeTrainerImplBase&) = delete;
    TreeTrainerImplBase& operator=(const TreeTrainerImplBase&) = delete;
};

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
class TreeTrainerImpl : public TreeTrainerImplBase 
{
public:
    TreeTrainerImpl(CRefXXf inData, CRefXs strata);
    virtual ~TreeTrainerImpl() = default;

    virtual unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const;

private:
    using RandomNumberEngine_ = splitmix;

    using BernoulliDistribution = typename std::conditional<
        sizeof(SampleIndex) == 8,
        FastBernoulliDistribution,
        VeryFastBernoulliDistribution
    >::type;

    vector<vector<SampleIndex>> createSortedSamples_() const;
    size_t initUsedSampleMask_(const StumpOptions& opt, CRefXd weights) const;
    size_t initUsedVariables_(const StumpOptions& opt) const;
    void initSortedUsedSamples_(size_t usedSampleCount, size_t j) const;
    pair<double, double> initSums_(const CRefXd& outData, const CRefXd& weights) const;

    const CRefXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const vector<vector<SampleIndex>> sortedSamples_;

    const CRefXs strata_;
    const size_t stratum0Count_;
    const size_t stratum1Count_;

    inline static thread_local RandomNumberEngine_ rne_;
    inline static thread_local vector<uint8_t> usedSampleMask_;
    inline static thread_local vector<size_t> usedVariables_;
    inline static thread_local vector<SampleIndex> tmpSamples_;
    inline static thread_local vector<SampleIndex> sortedUsedSamples_;

    inline static thread_local struct ThreadLocalInit_ {
        ThreadLocalInit_() {
            std::random_device rd;
            rne_.seed(rd);
        }
    } threadLocalInit_{};
};

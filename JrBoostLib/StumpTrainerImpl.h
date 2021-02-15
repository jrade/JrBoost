#pragma once

#include "../Tools/AGRandom.h"
#include "../Tools/BernoulliDistribution.h"

class StumpOptions;
class AbstractPredictor;


class StumpTrainerImplBase {
public:
    virtual ~StumpTrainerImplBase() = default;
    virtual unique_ptr<AbstractPredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const = 0;

protected:
    StumpTrainerImplBase() = default;

// deleted:
    StumpTrainerImplBase(const StumpTrainerImplBase&) = delete;
    StumpTrainerImplBase& operator=(const StumpTrainerImplBase&) = delete;
};

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
class StumpTrainerImpl : public StumpTrainerImplBase 
{
public:
    StumpTrainerImpl(CRefXXf inData, CRefXs strata);
    virtual ~StumpTrainerImpl() = default;

    virtual unique_ptr<AbstractPredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const;

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

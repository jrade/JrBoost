#pragma once

#include "../Tools/AGRandom.h"

class StumpOptions;
class AbstractPredictor;


class StumpTrainer {
public:
    StumpTrainer(CRefXXf inData, RefXs strata);
    ~StumpTrainer() = default;

    unique_ptr<AbstractPredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const;

// deleted:
    StumpTrainer() = delete;
    StumpTrainer(const StumpTrainer&) = delete;
    StumpTrainer& operator=(const StumpTrainer&) = delete;

private:
    vector<vector<SampleIndex>> sortSamples_() const;
    size_t initUsedSampleMask_(const StumpOptions& opt) const;
    size_t initUsedVariables_(const StumpOptions& opt) const;
    void initSums_(const CRefXd& outData, const CRefXd& weights) const;
    void initSortedUsedSamples_(size_t usedSampleCount, size_t j) const;

    const CRefXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const vector<vector<SampleIndex>> sortedSamples_;

    const RefXs strata_;
    const size_t stratum0Count_;
    const size_t stratum1Count_;

    using RandomNumberEngine_ = splitmix;

    inline static thread_local RandomNumberEngine_ rne_;
    inline static thread_local vector<char> usedSampleMask_;
    inline static thread_local vector<size_t> usedVariables_;
    inline static thread_local vector<SampleIndex> sortedUsedSamples_;
    inline static thread_local double sumW_;
    inline static thread_local double sumWY_;

    inline static thread_local struct ThreadLocalInit_ {
        ThreadLocalInit_() {
            std::random_device rd;
            rne_.seed(rd);
        }
    } threadLocalInit_{};
};

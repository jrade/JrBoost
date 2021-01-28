#pragma once

class StumpOptions;


class StumpTrainerShared {
public:
    StumpTrainerShared(CRefXXf inData, RefXs strata);
    ~StumpTrainerShared() = default;

    size_t initUsedSampleMask(vector<char>* usedSampleMask, const StumpOptions& opt, RandomNumberEngine& rne) const;
    void initSortedUsedSamples(
        vector<SampleIndex>* sortedUsedSamples, size_t usedSampleCount, const vector<char>& usedSampleMask, size_t j) const;

// deleted:
    StumpTrainerShared() = delete;
    StumpTrainerShared(const StumpTrainerShared&) = delete;
    StumpTrainerShared& operator=(const StumpTrainerShared&) = delete;

private:
    vector<vector<SampleIndex>> sortSamples_(CRefXXf inData) const;

private:
    const size_t sampleCount_;
    const size_t variableCount_;
    const vector<vector<SampleIndex>> sortedSamples_;

    const RefXs strata_;
    const size_t stratum0Count_;
    const size_t stratum1Count_;
};

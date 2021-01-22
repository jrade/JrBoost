#pragma once

#include "StumpOptions.h"

class StumpPredictor;

class StumpTrainer {
public:
    StumpTrainer(CRefXXf inData, RefXs strata); // stores references to the arrays
    ~StumpTrainer() = default;
    StumpPredictor train(CRefXd outData, CRefXd weights, const StumpOptions& options) const;

// deleted:
    StumpTrainer() = delete;
    StumpTrainer(const StumpTrainer&) = delete;
    StumpTrainer& operator=(const StumpTrainer&) = delete;

private:
    CRefXXf inData_;
    vector<vector<size_t>> sortedSamples_;

    RefXs strata_;
    size_t stratum0Count_;
    size_t stratum1Count_;

    mutable splitmix fastRNE_;
    mutable vector<char> usedSampleMask_;
    mutable vector<size_t> usedVariables_;
    mutable vector<size_t> sortedUsedSamples_;
};

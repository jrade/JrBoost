#pragma once

#include "StumpPredictor.h"
#include "StumpOptions.h"

class StumpPredictor;

class StumpTrainer {
public:
    StumpTrainer(CRefXXf inData, const ArrayXd& strata);
    ~StumpTrainer() = default;
    StumpPredictor train(const ArrayXd& outData, const ArrayXd& weights, const StumpOptions& options) const;

// deleted:
    StumpTrainer() = delete;
    StumpTrainer(const StumpTrainer&) = delete;
    StumpTrainer& operator=(const StumpTrainer&) = delete;

private:
    CRefXXf inData_;
    size_t sampleCount_;
    size_t variableCount_;
    vector<vector<size_t>> sortedSamples_;

    Eigen::Array<size_t, Eigen::Dynamic, 1> strata_;
    size_t stratum0Count_;
    size_t stratum1Count_;

    mutable splitmix fastRNE_{ std::random_device{} };
    mutable vector<char> usedSampleMask_;
    mutable vector<size_t> usedVariables_;
    mutable vector<size_t> sortedUsedSamples_;
};

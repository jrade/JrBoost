#pragma once

#include "AbstractTrainer.h"
#include "StumpPredictor.h"

class StumpOptions;

class StumpTrainer : public AbstractTrainer {
public:
    StumpTrainer();
    virtual ~StumpTrainer() = default;

    virtual void setInData(CRefXXf inData);
    virtual void setOutData(const ArrayXd& outData);
    virtual void setWeights(const ArrayXd& weights);
    virtual void setStrata(const ArrayXd& strata);
    virtual void setOptions(const AbstractOptions& opt);

    virtual AbstractPredictor* train() const;

private:
    CRefXXf inData_{ dummyArrayXXf };
    size_t sampleCount_{ 0 };
    size_t variableCount_{ 0 };
    vector<vector<size_t>> sortedSamples_;

    ArrayXd outData_;
    ArrayXd weights_;

    Eigen::Array<size_t, Eigen::Dynamic, 1> strata_;
    size_t stratum0Count_;
    size_t stratum1Count_;

    unique_ptr<StumpOptions> options_;

    // buffers used by train()
    mutable vector<char> usedSampleMask_;
    mutable vector<size_t> sortedUsedSamples_;
    mutable vector<size_t> usedVariables_;
};

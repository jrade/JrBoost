#pragma once

#include "AbstractTrainer.h"
#include "StumpPredictor.h"

class StumpOptions;

class StumpTrainer : public AbstractTrainer {
public:
    StumpTrainer();
    virtual ~StumpTrainer() = default;

    virtual void setInData(CRefXXf inData);
    virtual void setOutData(const ArrayXf& outData);
    virtual void setWeights(const ArrayXf& weights);
    virtual void setOptions(const AbstractOptions& opt);

    virtual AbstractPredictor* train() const;

private:
    template<typename F> AbstractPredictor*  trainImpl_() const;

    CRefXXf inData_{ dummyArrayXXf };
    size_t sampleCount_{ 0 };
    size_t variableCount_{ 0 };
    vector<vector<size_t>> sortedSamples_;

    ArrayXf outData_;
    ArrayXf weights_;
    unique_ptr<StumpOptions> options_;

    // buffers used by train()
    mutable vector<char> usedSampleMask_;
    mutable vector<size_t> sortedUsedSamples_;
    mutable vector<size_t> usedVariables_;
};

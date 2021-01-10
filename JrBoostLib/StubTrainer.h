#pragma once

#include "AbstractTrainer.h"
#include "StubPredictor.h"

class StubOptions;

class StubTrainer : public AbstractTrainer {
public:
    StubTrainer();
    virtual ~StubTrainer() = default;

    virtual void setInData(RefXXf inData);
    virtual void setOutData(const ArrayXf& outData);
    virtual void setWeights(const ArrayXf& weights);
    virtual void setOptions(const AbstractOptions& opt);

    virtual StubPredictor* train() const;

private:
    template<typename F> StubPredictor*  trainImpl_() const;

    RefXXf inData_{ dummyArrayXXf };
    size_t sampleCount_{ 0 };
    size_t variableCount_{ 0 };
    vector<vector<size_t>> sortedSamples_;

    ArrayXf outData_;
    ArrayXf weights_;
    unique_ptr<StubOptions> options_;

    // buffers used by train()
    mutable vector<char> usedSampleMask_;
    mutable vector<size_t> sortedUsedSamples_;
    mutable vector<size_t> usedVariables_;
};

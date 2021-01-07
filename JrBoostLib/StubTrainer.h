#pragma once

#include "AbstractTrainer.h"
#include "StubPredictor.h"

class StubOptions;

class StubTrainer : public AbstractTrainer {
public:
    StubTrainer() = default;
    virtual ~StubTrainer() = default;

    virtual void setInData(const ArrayXXf& inData);
    virtual void setOutData(const ArrayXf& outData) { outData_ = outData; }
    virtual void setWeights(const ArrayXf& weights) { weights_ = weights; }
    virtual void setOptions(const AbstractOptions& opt);

    virtual StubPredictor* train() const;

private:
    ArrayXf outData_;
    ArrayXf weights_;
    unique_ptr<StubOptions> options_;

    ArrayXXf inData_;
    int sampleCount_;
    int variableCount_;
    vector<vector<int>> sortedSamples_;

    // buffers used by train()
    mutable vector<char> usedSampleMask_;
    mutable vector<int> usedVariables_;
    mutable vector<int> sortedUsedSamples_;
};

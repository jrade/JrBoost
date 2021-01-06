#pragma once

#include "AbstractTrainer.h"
#include "StubPredictor.h"

class StubOptions;

class StubTrainer : public AbstractTrainer {
public:
    StubTrainer() = default;
    virtual ~StubTrainer() = default;

    virtual void setInData(const ArrayXXf& inData) { inData_ = inData; }
    virtual void setOutData(const ArrayXf& outData) { outData_ = outData; }
    virtual void setWeights(const ArrayXf& weights) { weights_ = weights; }
    virtual void setOptions(const AbstractOptions& opt);

    virtual StubPredictor* train() const;

private:
    ArrayXXf inData_;
    ArrayXf outData_;
    ArrayXf weights_;
    unique_ptr<StubOptions> options_;
};

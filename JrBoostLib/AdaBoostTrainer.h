#pragma once

#include "AbstractTrainer.h"
#include "BoostPredictor.h"

class AdaBoostOptions;

class AdaBoostTrainer : public AbstractTrainer {
public:
    AdaBoostTrainer();
    virtual ~AdaBoostTrainer() = default;

    virtual void setInData(RefXXf inData);
    virtual void setOutData(const ArrayXf& outData);
    virtual void setWeights(const ArrayXf& weights);
    virtual void setOptions(const AbstractOptions& opt);

    virtual BoostPredictor* train() const;

private:
    template<typename F>
    BoostPredictor* trainImpl_() const;

    ArrayXf outData_;
    ArrayXf weights_;
    unique_ptr<AdaBoostOptions> options_;

    RefXXf inData_{ dummyArrayXXf };
    size_t sampleCount_ = 0;
    size_t variableCount_ = 0;
};

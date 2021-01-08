#pragma once

class AbstractPredictor;
class AbstractOptions;

class AbstractTrainer {
protected:
    AbstractTrainer() = default;

public:
    virtual ~AbstractTrainer() = default;
    virtual void setInData(Eigen::Ref<ArrayXXf> inData) = 0;
    virtual void setOutData(const ArrayXf& outData) = 0;
    virtual void setWeights(const ArrayXf& weights) = 0;
    virtual void setOptions(const AbstractOptions& opt) = 0;
    virtual AbstractPredictor* train() const = 0;

// deleted:
    AbstractTrainer(const AbstractTrainer&) = delete;
    AbstractTrainer& operator=(const AbstractTrainer&) = delete;
    AbstractTrainer(AbstractTrainer&&) = delete;
    AbstractTrainer& operator=(AbstractTrainer&&) = delete;
};

#pragma once

class AbstractPredictor;
class AbstractOptions;

class AbstractTrainer {
protected:
    AbstractTrainer() = default;

public:
    virtual ~AbstractTrainer() = default;
    virtual void setInData(CRefXXf inData) = 0;
    virtual void setOutData(const ArrayXd& outData) = 0;
    virtual void setWeights(const ArrayXd& weights) = 0;
    virtual void setOptions(const AbstractOptions& opt) = 0;
    virtual AbstractPredictor* train() const = 0;

// deleted:
    AbstractTrainer(const AbstractTrainer&) = delete;
    AbstractTrainer& operator=(const AbstractTrainer&) = delete;
    AbstractTrainer(AbstractTrainer&&) = delete;
    AbstractTrainer& operator=(AbstractTrainer&&) = delete;
};

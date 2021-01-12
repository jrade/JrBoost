#pragma once

#include "AbstractOptions.h"
#include "AdaBoostTrainer.h"

class AdaBoostOptions : public AbstractOptions {
public:
    AdaBoostOptions() = default;
    virtual ~AdaBoostOptions() = default;

    size_t iterationCount() const;
    float eta() const;
    bool highPrecision() const;
    AbstractOptions* baseOptions() const;

    void setIterationCount(size_t n);
    void setEta(float eta);
    void setHighPrecision(bool b);
    void setBaseOptions(const AbstractOptions& opt);

    virtual AdaBoostOptions* clone() const;
    virtual AdaBoostTrainer* createTrainer() const;

private:
    AdaBoostOptions(const AdaBoostOptions&);

    size_t iterationCount_{ 100 };
    float eta_{ 0.3f };
    bool highPrecision_{ true };
    unique_ptr<AbstractOptions> baseOptions_;
};

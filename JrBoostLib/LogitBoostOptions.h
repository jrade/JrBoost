#pragma once

#include "AbstractOptions.h"
#include "LogitBoostTrainer.h"

class LogitBoostOptions : public AbstractOptions {
public:
    LogitBoostOptions() = default;
    virtual ~LogitBoostOptions() = default;

    size_t iterationCount() const;
    float eta() const;
    bool highPrecision() const;
    AbstractOptions* baseOptions() const;

    void setIterationCount(size_t n);
    void setEta(float eta);
    void setHighPrecision(bool b);
    void setBaseOptions(const AbstractOptions& opt);

    virtual LogitBoostOptions* clone() const;
    virtual LogitBoostTrainer* createTrainer() const;

private:
    LogitBoostOptions(const LogitBoostOptions&);

    size_t iterationCount_{ 100 };
    float eta_{ 0.3f };
    bool highPrecision_{ true };
    unique_ptr<AbstractOptions> baseOptions_;
};

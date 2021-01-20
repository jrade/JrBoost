#pragma once

#include "AbstractOptions.h"
#include "LogitBoostTrainer.h"

class LogitBoostOptions : public AbstractOptions {
public:
    LogitBoostOptions() = default;
    virtual ~LogitBoostOptions() = default;

    vector<size_t> iterationCount() const { return iterationCount_; }
    vector<double> eta() const { return eta_; }
    size_t logStep() const { return logStep_; }
    AbstractOptions* baseOptions() const { return baseOptions_->clone(); }

    void setIterationCount(const vector<size_t>& n);
    void setEta(const vector<double>& eta);
    void setLogStep(size_t n);
    void setBaseOptions(const AbstractOptions& opt);

    virtual LogitBoostOptions* clone() const;
    virtual LogitBoostTrainer* createTrainer() const;

private:
    LogitBoostOptions(const LogitBoostOptions&);

    vector<size_t> iterationCount_{ 100 };
    vector<double> eta_{ 0.3f };
    size_t logStep_{ 0 };
    unique_ptr<AbstractOptions> baseOptions_;
};
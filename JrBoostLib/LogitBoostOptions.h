#pragma once

#include "AbstractOptions.h"
#include "LogitBoostTrainer.h"

class LogitBoostOptions : public AbstractOptions {
public:
    LogitBoostOptions() = default;
    virtual ~LogitBoostOptions() = default;

    vector<size_t> iterationCount() const;
    vector<double> eta() const;
    bool highPrecision() const;
    AbstractOptions* baseOptions() const;

    void setIterationCount(const vector<size_t>& n);
    void setEta(const vector<double>& eta);
    void setHighPrecision(bool b);
    void setBaseOptions(const AbstractOptions& opt);

    virtual LogitBoostOptions* clone() const;
    virtual LogitBoostTrainer* createTrainer() const;

private:
    LogitBoostOptions(const LogitBoostOptions&);

    vector<size_t> iterationCount_{ 100 };
    vector<double> eta_{ 0.3f };
    bool highPrecision_{ true };
    unique_ptr<AbstractOptions> baseOptions_;
};

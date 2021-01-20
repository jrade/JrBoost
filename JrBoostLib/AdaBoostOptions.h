#pragma once

#include "AbstractOptions.h"
#include "AdaBoostTrainer.h"

class AdaBoostOptions : public AbstractOptions {
public:
    AdaBoostOptions() = default;
    virtual ~AdaBoostOptions() = default;

    vector<size_t> iterationCount() const;
    vector<double> eta() const;
    bool highPrecision() const;
    AbstractOptions* baseOptions() const;

    void setIterationCount(const vector<size_t>& n);
    void setEta(const vector<double>& eta);
    void setHighPrecision(bool b);
    void setBaseOptions(const AbstractOptions& opt);

    virtual AdaBoostOptions* clone() const;
    virtual AdaBoostTrainer* createTrainer() const;

private:
    AdaBoostOptions(const AdaBoostOptions&);

    vector<size_t> iterationCount_{ 100 };
    vector<double> eta_{ 0.3f };
    bool highPrecision_{ true };
    unique_ptr<AbstractOptions> baseOptions_;
};

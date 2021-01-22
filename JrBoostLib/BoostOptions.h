#pragma once

#include "StumpOptions.h"


class BoostOptions {
public:
    BoostOptions() = default;
    BoostOptions(const BoostOptions&) = default;
    BoostOptions& operator=(const BoostOptions&) = default;
    ~BoostOptions() = default;

    size_t iterationCount() const { return iterationCount_; }
    double eta() const { return eta_; }
    size_t logStep() const { return logStep_; }
    const StumpOptions& baseOptions() const { return baseOptions_; }

    void setIterationCount(size_t n);
    void setEta(double eta);
    void setLogStep(size_t n);
    void setBaseOptions(const StumpOptions& opt);

private:
    size_t iterationCount_{ 1000 };
    double eta_{ 0.1 };
    size_t logStep_{ 0 };
    StumpOptions baseOptions_{};
};

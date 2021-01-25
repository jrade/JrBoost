#pragma once

#include "StumpOptions.h"


class BoostOptions {
public:
    enum class Method { Ada, Logit };

    BoostOptions() = default;
    BoostOptions(const BoostOptions&) = default;
    BoostOptions& operator=(const BoostOptions&) = default;
    ~BoostOptions() = default;

    size_t iterationCount() const { return iterationCount_; }
    double eta() const { return eta_; }
    size_t logStep() const { return logStep_; }
    Method method() const { return method_; }

    void setMethod(Method m);
    void setIterationCount(size_t n);
    void setEta(double eta);
    void setLogStep(size_t n);

    StumpOptions& base() { return base_; }
    const StumpOptions& base() const { return base_; }

private:
    Method method_{ Method::Ada };
    size_t iterationCount_{ 1000 };
    double eta_{ 0.1 };
    size_t logStep_{ 0 };
    StumpOptions base_{};
};

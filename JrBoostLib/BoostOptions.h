#pragma once

#include "StumpOptions.h"


class BoostOptions : public StumpOptions {
public:
    enum class Method { Ada, Logit };

    BoostOptions() = default;
    BoostOptions(const BoostOptions&) = default;
    BoostOptions& operator=(const BoostOptions&) = default;
    ~BoostOptions() = default;

    Method method() const { return method_; }
    size_t iterationCount() const { return iterationCount_; }
    double eta() const { return eta_; }

    void setMethod(Method m);
    void setIterationCount(size_t n);
    void setEta(double eta);

    double cost() const;

private:
    Method method_{ Method::Ada };
    size_t iterationCount_{ 1000 };
    double eta_{ 0.3 };
};

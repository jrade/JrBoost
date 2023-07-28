//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BaseOptions.h"


class BoostOptions : public BaseOptions {   // POD class, so no need for virtual destructor
public:
    BoostOptions() = default;
    BoostOptions(const BoostOptions&) = default;
    BoostOptions& operator=(const BoostOptions&) = default;
    ~BoostOptions() = default;

    double gamma() const { return gamma_; }
    size_t iterationCount() const { return iterationCount_; }
    double eta() const { return eta_; }
    bool fastExp() const { return fastExp_; }

    void setGamma(double gamma);
    void setIterationCount(size_t n);
    void setEta(double eta);
    void setFastExp(bool b);

private:
    double gamma_{1.0};
    size_t iterationCount_{1000};
    double eta_{0.1};
    bool fastExp_{true};
};

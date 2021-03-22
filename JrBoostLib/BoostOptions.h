//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "StumpOptions.h"


class BoostOptions : public StumpOptions {
public:
    static const int Ada = 0;
    static const int Alpha = 1;

    BoostOptions() = default;
    BoostOptions(const BoostOptions&) = default;
    BoostOptions& operator=(const BoostOptions&) = default;
    ~BoostOptions() = default;

    int method() const { return method_; }
    double alpha() const { return alpha_; }
    size_t iterationCount() const { return iterationCount_; }
    double eta() const { return eta_; }
    double minAbsSampleWeight() const { return minAbsSampleWeight_; }
    double minRelSampleWeight() const { return minRelSampleWeight_; }

    void setMethod(int m);
    void setAlpha(double alpha);
    void setIterationCount(size_t n);
    void setEta(double eta);
    void setMinAbsSampleWeight(double w);
    void setMinRelSampleWeight(double w);

    double cost() const;

private:
    int method_{ Ada };
    double alpha_{ 0.5 };
    size_t iterationCount_{ 1000 };
    double eta_{ 0.3 };
    double minAbsSampleWeight_{ 0.0 };
    double minRelSampleWeight_{ 0.0 };
};

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostOptions.h"


void BoostOptions::setMethod(int m)
{
    ASSERT(m == Ada || m == Logit);
    method_ = m;
}

void BoostOptions::setGamma(double gamma)
{
    ASSERT(0.0 <= gamma && gamma <= 1.0);
    gamma_ = gamma;
}

void BoostOptions::setIterationCount(size_t n)
{
    iterationCount_ = n;
}

void BoostOptions::setEta(double eta)
{
    ASSERT(eta > 0.0);
    eta_ = eta;
}

void BoostOptions::setMinAbsSampleWeight(double w)
{
    ASSERT(w >= 0.0);
    minAbsSampleWeight_ = w;
}

void BoostOptions::setMinRelSampleWeight(double w)
{
    ASSERT(0.0 <= w && w <= 1.0);
    minRelSampleWeight_ = w;
}

void BoostOptions::setFastExp(bool b)
{
    fastExp_ = b;
}

double BoostOptions::cost() const
{
    return StumpOptions::cost() * iterationCount_;
}
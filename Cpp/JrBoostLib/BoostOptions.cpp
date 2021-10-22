//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostOptions.h"


void BoostOptions::setGamma(double gamma)
{
    if (!(gamma >= 0.0 && gamma <= 1.0))        // carefully written to trap NaN
        throw std::invalid_argument("gamma must lie in the interval [0.0, 1.0].");
    gamma_ = gamma;
}

void BoostOptions::setIterationCount(size_t n)
{
    iterationCount_ = n;
}

void BoostOptions::setEta(double eta)
{
    if (!(eta > 0.0))       // carefully written to trap NaN
        throw std::invalid_argument("eta must be positive.");
    eta_ = eta;
}

void BoostOptions::setFastExp(bool b)
{
    fastExp_ = b;
}

double BoostOptions::cost() const
{
    return TreeOptions::cost() * iterationCount_ / eta_;
}

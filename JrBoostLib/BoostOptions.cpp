//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BoostOptions.h"


void BoostOptions::setMethod(Method m)
{
    method_ = m;
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

double BoostOptions::cost() const
{
    return StumpOptions::cost() * iterationCount_;
}

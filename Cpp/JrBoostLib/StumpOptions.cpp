//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "StumpOptions.h"


void StumpOptions::setMaxDepth(size_t d)
{
    if (d <= 0)
        throw std::invalid_argument("maxDepth must be positive.");
    maxDepth_ = d;
}

void StumpOptions::setUsedSampleRatio(double r)
{
    if (!(r >= 0.0 && r <= 1.0))
        throw std::invalid_argument("usedSampleRatio must lie in the interval [0.0, 1.0].");
    usedSampleRatio_ = r;
}

void StumpOptions::setUsedVariableRatio(double r)
{
    if (!(r >= 0.0 && r <= 1.0))
        throw std::invalid_argument("usedVariableRatio must lie in the interval [0.0, 1.0].");
    usedVariableRatio_ = r;
}

void StumpOptions::setTopVariableCount(size_t n)
{
    if (n <= 0)
        throw std::invalid_argument("topVariableCount must be positive.");
    topVariableCount_ = n;
}

void StumpOptions::setMinAbsSampleWeight(double w)
{
    if (!(w >= 0.0))
        throw std::invalid_argument("minAbsSampleWeight must be non-negative.");
    minAbsSampleWeight_ = w;
}

void StumpOptions::setMinRelSampleWeight(double w)
{
    if (!(w >= 0.0 && w <= 1.0))
        throw std::invalid_argument("minRelSampleWeight must lie in the interval [0.0, 1.0].");
    minRelSampleWeight_ = w;
}

void StumpOptions::setMinNodeSize(size_t n)
{
    if (n <= 0)
        throw std::invalid_argument("minNodeSize must be positive.");
    minNodeSize_ = n;
}

void StumpOptions::setMinNodeWeight(double w)
{
    if (!(w >= 0))
        throw std::invalid_argument("minNodeWeight must be nonnegative.");
    minNodeWeight_ = w;
}

void StumpOptions::setIsStratified(bool b)
{
    isStratified_ = b;
}

double StumpOptions::cost() const
{
    return (1.0 + 4.0 * usedSampleRatio_) * usedVariableRatio_ * topVariableCount_;
}

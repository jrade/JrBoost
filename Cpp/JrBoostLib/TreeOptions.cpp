//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeOptions.h"


void TreeOptions::setMaxDepth(size_t d)
{
    if (d < 0)
        throw std::invalid_argument("maxDepth must be non-negative.");
    maxDepth_ = d;
}

void TreeOptions::setUsedSampleRatio(double r)
{
    if (!(r >= 0.0 && r <= 1.0))
        throw std::invalid_argument("usedSampleRatio must lie in the interval [0.0, 1.0].");
    usedSampleRatio_ = r;
}

void TreeOptions::setUsedVariableRatio(double r)
{
    if (!(r >= 0.0 && r <= 1.0))
        throw std::invalid_argument("usedVariableRatio must lie in the interval [0.0, 1.0].");
    usedVariableRatio_ = r;
}

void TreeOptions::setTopVariableCount(size_t n)
{
    if (n <= 0)
        throw std::invalid_argument("topVariableCount must be positive.");
    topVariableCount_ = n;
}

void TreeOptions::setMinAbsSampleWeight(double w)
{
    if (!(w >= 0.0))
        throw std::invalid_argument("minAbsSampleWeight must be non-negative.");
    minAbsSampleWeight_ = w;
}

void TreeOptions::setMinRelSampleWeight(double w)
{
    if (!(w >= 0.0 && w <= 1.0))
        throw std::invalid_argument("minRelSampleWeight must lie in the interval [0.0, 1.0].");
    minRelSampleWeight_ = w;
}

void TreeOptions::setMinNodeSize(size_t n)
{
    if (n <= 0)
        throw std::invalid_argument("minNodeSize must be positive.");
    minNodeSize_ = n;
}

void TreeOptions::setMinNodeWeight(double w)
{
    if (!(w >= 0))
        throw std::invalid_argument("minNodeWeight must be nonnegative.");
    minNodeWeight_ = w;
}

void TreeOptions::setMinGain(double g)
{
    if (!(g >= 0))
        throw std::invalid_argument("minGain must be nonnegative.");
    minGain_ = g;
}

void TreeOptions::setIsStratified(bool b)
{
    isStratified_ = b;
}

void TreeOptions::setPruneFactor(double p)
{
    if (!(p >= 0.0 && p <= 1.0))
        throw std::invalid_argument("pruneFactor must lie in the interval [0.0, 1.0].");
    pruneFactor_ = p;
}

void TreeOptions::setSaveMemory(bool b)
{
    saveMemory_ = b;
}

//----------------------------------------------------------------------------------------------------------------------

double TreeOptions::cost() const
{
    return usedVariableRatio_ * topVariableCount_ * usedSampleRatio_ * maxDepth_;
}

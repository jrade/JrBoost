//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "BaseOptions.h"


void BaseOptions::setForestSize(size_t n)
{
    if (n == 0)
        throw std::invalid_argument("forestSize must be positive.");
    forestSize_ = n;
}

void BaseOptions::setMaxTreeDepth(size_t d) { maxTreeDepth_ = d; }

void BaseOptions::setMinAbsSampleWeight(double w)
{
    if (!(w >= 0.0))   // carefully written to trap NaN
        throw std::invalid_argument("minAbsSampleWeight must be non-negative.");
    minAbsSampleWeight_ = w;
}

void BaseOptions::setMinRelSampleWeight(double w)
{
    if (!(w >= 0.0 && w <= 1.0))   // carefully written to trap NaN
        throw std::invalid_argument("minRelSampleWeight must lie in the interval [0.0, 1.0].");
    minRelSampleWeight_ = w;
}

void BaseOptions::setUsedSampleRatio(double r)
{
    if (!(r > 0.0 && r <= 1.0))   // carefully written to trap NaN
        throw std::invalid_argument("usedSampleRatio must lie in the interval (0.0, 1.0].");
    usedSampleRatio_ = r;
}

void BaseOptions::setStratifiedSamples(bool b) { stratifiedSamples_ = b; }

void BaseOptions::setTopVariableCount(size_t n)
{
    if (n == 0)
        throw std::invalid_argument("topVariableCount must be positive.");
    topVariableCount_ = n;
}

void BaseOptions::setUsedVariableRatio(double r)
{
    if (!(r >= 0.0 && r <= 1.0))   // carefully written to trap NaN
        throw std::invalid_argument("usedVariableRatio must lie in the interval [0.0, 1.0].");
    usedVariableRatio_ = r;
}

void BaseOptions::setSelectVariablesByLevel(bool b) { selectVariablesByLevel_ = b; }

void BaseOptions::setMinNodeSize(size_t n)
{
    if (n == 0)
        throw std::invalid_argument("minNodeSize must be positive.");
    minNodeSize_ = n;
}

void BaseOptions::setMinNodeWeight(double w)
{
    if (!(w >= 0))   // carefully written to trap NaN
        throw std::invalid_argument("minNodeWeight must be nonnegative.");
    minNodeWeight_ = w;
}

void BaseOptions::setMinNodeGain(double g)
{
    if (!(g >= 0))   // carefully written to trap NaN
        throw std::invalid_argument("minNodeGain must be nonnegative.");
    minNodeGain_ = g;
}

void BaseOptions::setPruneFactor(double p)
{
    if (!(p >= 0.0 && p <= 1.0))   // carefully written to trap NaN
        throw std::invalid_argument("pruneFactor must lie in the interval [0.0, 1.0].");
    pruneFactor_ = p;
}

void BaseOptions::setSaveMemory(bool b) { saveMemory_ = b; }

void BaseOptions::setTest(size_t n) { test_ = n; }

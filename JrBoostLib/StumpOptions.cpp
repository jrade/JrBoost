#include "pch.h"
#include "StumpOptions.h"


void StumpOptions::setUsedSampleRatio(double r)
{
    ASSERT(r > 0.0 && r <= 1.0);
    usedSampleRatio_ = r;
}

void StumpOptions::setUsedVariableRatio(double r)
{
    ASSERT(r > 0.0 && r <= 1.0);
    usedVariableRatio_ = r;
}

void StumpOptions::setTopVariableCount(size_t n)
{
    ASSERT(n != 0);
    topVariableCount_ = n;
}

void StumpOptions::setMinSampleWeight(double w)
{
    ASSERT(w >= 0.0);
    minSampleWeight_ = w;
}

void StumpOptions::setMinNodeSize(size_t n)
{
    ASSERT(n != 0);
    minNodeSize_ = n;
}

void StumpOptions::setMinNodeWeight(double w)
{
    minNodeWeight_ = w;
}

void StumpOptions::setIsStratified(bool b)
{
    isStratified_ = b;
}

double StumpOptions::cost() const
{
    return usedSampleRatio_ * usedVariableRatio_ * topVariableCount_;
}

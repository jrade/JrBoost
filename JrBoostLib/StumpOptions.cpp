#include "pch.h"
#include "StumpOptions.h"

double StumpOptions::usedSampleRatio() const
{
    return usedSampleRatio_;
}

double StumpOptions::usedVariableRatio() const
{
    return usedVariableRatio_;
}

size_t StumpOptions::minNodeSize() const
{
    return minNodeSize_;
}

double StumpOptions::minNodeWeight() const
{
    return minNodeWeight_;
}

bool StumpOptions::isStratified() const
{
    return isStratified_;
}

bool StumpOptions::profile() const {
    return profile_; 
}

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

void StumpOptions::setIsStratified(bool b)
{
    isStratified_ = b;
}

void StumpOptions::setMinNodeSize(size_t n)
{
    ASSERT(n > 0);
    minNodeSize_ = n;
}

void StumpOptions::setMinNodeWeight(double w)
{
    minNodeWeight_ = w;
}

void StumpOptions::setProfile(bool p)
{ 
    profile_ = p;
}

StumpOptions* StumpOptions::clone() const
{
    return new StumpOptions{ *this };
}

StumpTrainer* StumpOptions::createTrainer() const
{
    StumpTrainer* trainer = new StumpTrainer;
    trainer->setOptions(*this);
    return trainer;
}

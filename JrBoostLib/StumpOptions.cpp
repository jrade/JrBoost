#include "pch.h"
#include "StumpOptions.h"

float StumpOptions::usedSampleRatio() const
{
    return usedSampleRatio_;
}

float StumpOptions::usedVariableRatio() const
{
    return usedVariableRatio_;
}

size_t StumpOptions::minNodeSize() const
{
    return minNodeSize_;
}

float StumpOptions::minNodeWeight() const
{
    return minNodeWeight_;
}

bool StumpOptions::isStratified() const
{
    return isStratified_;
}

bool StumpOptions::highPrecision() const
{
    return highPrecision_;
}

bool StumpOptions::profile() const {
    return profile_; 
}

void StumpOptions::setUsedSampleRatio(float r)
{
    ASSERT(r > 0.0f && r <= 1.0f);
    usedSampleRatio_ = r;
}

void StumpOptions::setUsedVariableRatio(float r)
{
    ASSERT(r > 0.0f && r <= 1.0f);
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

void StumpOptions::setMinNodeWeight(float w)
{
    minNodeWeight_ = w;
}

void StumpOptions::setHighPrecision(bool p)
{
    highPrecision_ = p;
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

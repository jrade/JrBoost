#include "pch.h"
#include "StubOptions.h"

void StubOptions::setUsedSampleRatio(float r)
{
    ASSERT(r > 0.0f && r <= 1.0f);
    usedSampleRatio_ = r;
}

void StubOptions::setUsedVariableRatio(float r)
{
    ASSERT(r > 0.0f && r <= 1.0f);
    usedVariableRatio_ = r;
}

StubOptions* StubOptions::clone() const
{
    return new StubOptions(*this);
}

StubTrainer* StubOptions::createTrainer() const
{
    StubTrainer* trainer = new StubTrainer;
    trainer->setOptions(*this);
    return trainer;
}

#include "pch.h"
#include "StubOptions.h"

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

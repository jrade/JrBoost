#include "pch.h"
#include "StubTrainer.h"
#include "StubOptions.h"

void StubTrainer::setOptions(const AbstractOptions& opt)
{
    options_.reset(dynamic_cast<const StubOptions&>(opt).clone());
}

StubPredictor* StubTrainer::train() const
{
    int variableCount = static_cast<int>(inData_.cols());
    return new StubPredictor(variableCount);
};

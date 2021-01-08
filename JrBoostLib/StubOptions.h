#pragma once

#include "AbstractOptions.h"
#include "StubTrainer.h"

class StubOptions : public AbstractOptions {
public:   
    StubOptions() = default;
    virtual ~StubOptions() = default;

    virtual StubOptions* clone() const;
    virtual StubTrainer* createTrainer() const;

    float usedSampleRatio = 1.0f;
    float usedVariableRatio = 1.0f;
    string precision = "single";

private:
    StubOptions(const StubOptions&) = default;
};

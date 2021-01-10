#pragma once

#include "AbstractOptions.h"
#include "StubTrainer.h"

class StubOptions : public AbstractOptions {
public:   
    StubOptions() = default;
    virtual ~StubOptions() = default;

    virtual StubOptions* clone() const;
    virtual StubTrainer* createTrainer() const;

    float usedSampleRatio() const;
    float usedVariableRatio() const;
    bool highPrecision() const;
    bool profile() const;

    void setUsedSampleRatio(float r);
    void setUsedVariableRatio(float r);
    void setHighPrecision(bool p);
    void setProfile(bool p);

private:
    StubOptions(const StubOptions&) = default;

    float usedSampleRatio_{ 1.0f };
    float usedVariableRatio_{ 1.0f };
    bool highPrecision_{ false };
    bool profile_{ false };
};

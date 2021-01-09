#pragma once

#include "AbstractOptions.h"
#include "StubTrainer.h"

class StubOptions : public AbstractOptions {
public:   
    StubOptions() = default;
    virtual ~StubOptions() = default;

    virtual StubOptions* clone() const;
    virtual StubTrainer* createTrainer() const;

    float usedSampleRatio() const { return usedSampleRatio_; }
    float usedVariableRatio() const { return usedVariableRatio_; }
    bool highPrecision() const { return highPrecision_; }
    bool profile() const { return profile_; }

    void setUsedSampleRatio(float r);
    void setUsedVariableRatio(float r);
    void setHighPrecision(bool p) { highPrecision_ = p; }
    void setProfile(bool p) { profile_ = p; }

private:
    StubOptions(const StubOptions&) = default;

    float usedSampleRatio_{ 1.0f };
    float usedVariableRatio_{ 1.0f };
    bool highPrecision_{ false };
    bool profile_{ false };
};

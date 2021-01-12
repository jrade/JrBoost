#pragma once

#include "AbstractOptions.h"
#include "StumpTrainer.h"

class StumpOptions : public AbstractOptions {
public:   
    StumpOptions() = default;
    virtual ~StumpOptions() = default;

    virtual StumpOptions* clone() const;
    virtual StumpTrainer* createTrainer() const;

    float usedSampleRatio() const;
    float usedVariableRatio() const;
    bool highPrecision() const;
    bool profile() const;

    void setUsedSampleRatio(float r);
    void setUsedVariableRatio(float r);
    void setHighPrecision(bool p);
    void setProfile(bool p);

private:
    StumpOptions(const StumpOptions&) = default;

    float usedSampleRatio_{ 1.0f };
    float usedVariableRatio_{ 1.0f };
    bool highPrecision_{ false };
    bool profile_{ false };
};

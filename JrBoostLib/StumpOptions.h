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
    size_t minNodeSize() const;
    float minNodeWeight() const;
    bool isStratified() const;
    bool highPrecision() const;
    bool profile() const;

    void setUsedSampleRatio(float r);
    void setUsedVariableRatio(float r);
    void setMinNodeSize(size_t n);
    void setMinNodeWeight(float w);
    void setIsStratified(bool b);
    void setHighPrecision(bool p);
    void setProfile(bool p);

private:
    StumpOptions(const StumpOptions&) = default;

    float usedSampleRatio_{ 1.0f };
    float usedVariableRatio_{ 1.0f };
    size_t minNodeSize_{ 1 };
    float minNodeWeight_{ 0.0f };
    bool isStratified_{ false };
    bool highPrecision_{ false };
    bool profile_{ false };
};

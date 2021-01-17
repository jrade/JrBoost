#pragma once

#include "AbstractOptions.h"
#include "StumpTrainer.h"

class StumpOptions : public AbstractOptions {
public:   
    StumpOptions() = default;
    virtual ~StumpOptions() = default;

    virtual StumpOptions* clone() const;
    virtual StumpTrainer* createTrainer() const;

    double usedSampleRatio() const;
    double usedVariableRatio() const;
    size_t minNodeSize() const;
    double minNodeWeight() const;
    bool isStratified() const;
    bool profile() const;

    void setUsedSampleRatio(double r);
    void setUsedVariableRatio(double r);
    void setMinNodeSize(size_t n);
    void setMinNodeWeight(double w);
    void setIsStratified(bool b);
    void setProfile(bool p);

private:
    StumpOptions(const StumpOptions&) = default;

    double usedSampleRatio_{ 1.0 };
    double usedVariableRatio_{ 1.0 };
    size_t minNodeSize_{ 1 };
    double minNodeWeight_{ 0.0 };
    bool isStratified_{ false };
    bool profile_{ false };
};

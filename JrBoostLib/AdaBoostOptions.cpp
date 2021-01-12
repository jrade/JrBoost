#include "pch.h"
#include "AdaBoostOptions.h"

size_t AdaBoostOptions::iterationCount() const
{
    return iterationCount_;
}

float AdaBoostOptions::eta() const
{
    return eta_;
}

bool AdaBoostOptions::highPrecision() const
{
    return highPrecision_;
}

AbstractOptions* AdaBoostOptions::baseOptions() const
{
    return baseOptions_->clone();
}

void AdaBoostOptions::setIterationCount(size_t n)
{
    iterationCount_ = n;
}

void AdaBoostOptions::setEta(float eta)
{
    ASSERT(eta > 0.0f);
    eta_ = eta;
}

void AdaBoostOptions::setHighPrecision(bool b)
{
    highPrecision_ = b;
}

void AdaBoostOptions::setBaseOptions(const AbstractOptions& opt)
{
    baseOptions_.reset(opt.clone());
}

AdaBoostOptions* AdaBoostOptions::clone() const
{
    return new AdaBoostOptions{ *this };
}

AdaBoostTrainer* AdaBoostOptions::createTrainer() const
{
    AdaBoostTrainer* trainer = new AdaBoostTrainer;
    trainer->setOptions(*this);
    return trainer;
}

AdaBoostOptions::AdaBoostOptions(const AdaBoostOptions& a) :
    AbstractOptions{ a },
    iterationCount_{ a.iterationCount_ },
    eta_{ a.eta_ },
    highPrecision_{ a.highPrecision_ },
    baseOptions_{ a.baseOptions_->clone() }
{}

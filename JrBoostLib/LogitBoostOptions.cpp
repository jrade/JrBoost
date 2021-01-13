#include "pch.h"
#include "LogitBoostOptions.h"

size_t LogitBoostOptions::iterationCount() const
{
    return iterationCount_;
}

float LogitBoostOptions::eta() const
{
    return eta_;
}

bool LogitBoostOptions::highPrecision() const
{
    return highPrecision_;
}

AbstractOptions* LogitBoostOptions::baseOptions() const
{
    return baseOptions_->clone();
}

void LogitBoostOptions::setIterationCount(size_t n)
{
    iterationCount_ = n;
}

void LogitBoostOptions::setEta(float eta)
{
    ASSERT(eta > 0.0f);
    eta_ = eta;
}

void LogitBoostOptions::setHighPrecision(bool b)
{
    highPrecision_ = b;
}

void LogitBoostOptions::setBaseOptions(const AbstractOptions& opt)
{
    baseOptions_.reset(opt.clone());
}

LogitBoostOptions* LogitBoostOptions::clone() const
{
    return new LogitBoostOptions{ *this };
}

LogitBoostTrainer* LogitBoostOptions::createTrainer() const
{
    LogitBoostTrainer* trainer = new LogitBoostTrainer;
    trainer->setOptions(*this);
    return trainer;
}

LogitBoostOptions::LogitBoostOptions(const LogitBoostOptions& a) :
    AbstractOptions{ a },
    iterationCount_{ a.iterationCount_ },
    eta_{ a.eta_ },
    highPrecision_{ a.highPrecision_ },
    baseOptions_{ a.baseOptions_->clone() }
{}

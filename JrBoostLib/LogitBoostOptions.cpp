#include "pch.h"
#include "LogitBoostOptions.h"


void LogitBoostOptions::setIterationCount(const vector<size_t>& n)
{
    iterationCount_ = n;
}

void LogitBoostOptions::setEta(const vector<double>& eta)
{
    ASSERT(eta.empty() || *std::min_element(begin(eta), end(eta)) > 0.0);
    eta_ = eta;
}

void LogitBoostOptions::setLogStep(size_t n)
{
    logStep_ = n;
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
    logStep_{ a.logStep_ },
    baseOptions_{ a.baseOptions_->clone() }
{}

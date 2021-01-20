#include "pch.h"
#include "AdaBoostOptions.h"


void AdaBoostOptions::setIterationCount(const vector<size_t>& n)
{
    iterationCount_ = n;
}

void AdaBoostOptions::setEta(const vector<double>& eta)
{
    ASSERT(eta.empty()  || *std::min_element(begin(eta), end(eta)) > 0.0);
    eta_ = eta;
}

void AdaBoostOptions::setLogStep(size_t n)
{
    logStep_ = n;
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
    logStep_{ a.logStep_ },
    baseOptions_{ a.baseOptions_->clone() }
{}

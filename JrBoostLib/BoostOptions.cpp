#include "pch.h"
#include "BoostOptions.h"


void BoostOptions::setIterationCount(size_t n)
{
    iterationCount_ = n;
}

void BoostOptions::setEta(double eta)
{
    ASSERT(eta > 0.0);
    eta_ = eta;
}

void BoostOptions::setLogStep(size_t n)
{
    logStep_ = n;
}

void BoostOptions::setBaseOptions(const StumpOptions& opt)
{
    baseOptions_ = opt;
}

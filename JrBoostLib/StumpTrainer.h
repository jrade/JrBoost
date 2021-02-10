#pragma once

#include "StumpTrainerImpl.h"


class StumpTrainer
{
public:
    StumpTrainer(CRefXXf inData, RefXs strata) :
        impl_(createImpl(inData, strata))
    {
    }

    static unique_ptr<StumpTrainerImplBase> createImpl(CRefXXf inData, RefXs strata)
    {
        size_t sampleCount = static_cast<size_t>(inData.rows());
        if (sampleCount < 0x100)
            return std::make_unique<StumpTrainerImpl<uint8_t>>(inData, strata);
        else if (sampleCount < 0x10000)
            return std::make_unique<StumpTrainerImpl<uint16_t>>(inData, strata);
        else if (sampleCount < 0x100000000)
            return std::make_unique<StumpTrainerImpl<uint32_t>>(inData, strata);
        else
            return std::make_unique<StumpTrainerImpl<uint64_t>>(inData, strata);
    }

    ~StumpTrainer() = default;

    virtual unique_ptr<AbstractPredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const
    {
        return impl_->train(outData, weights, options);
    }

private:
    const unique_ptr<StumpTrainerImplBase> impl_;
};

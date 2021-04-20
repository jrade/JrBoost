//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "StumpTrainer.h"
#include "StumpTrainerImpl.h"
#include "BasePredictor.h"


StumpTrainer::StumpTrainer(CRefXXf inData, CRefXs strata) :
    impl_(createImpl_(inData, strata))
{
}

shared_ptr<StumpTrainerImplBase> StumpTrainer::createImpl_(CRefXXf inData, CRefXs strata)
{
    size_t sampleCount = static_cast<size_t>(inData.rows());
    if (sampleCount < 0x100)
        return std::make_shared<StumpTrainerImpl<uint8_t>>(inData, strata);
    else if (sampleCount < 0x10000)
        return std::make_shared<StumpTrainerImpl<uint16_t>>(inData, strata);
    else if (sampleCount < 0x100000000)
        return std::make_shared<StumpTrainerImpl<uint32_t>>(inData, strata);
    else
        return std::make_shared<StumpTrainerImpl<uint64_t>>(inData, strata);
}

unique_ptr<BasePredictor> StumpTrainer::train(CRefXd outData, CRefXd weights, const StumpOptions& options) const
{
    return impl_->train(outData, weights, options);
}
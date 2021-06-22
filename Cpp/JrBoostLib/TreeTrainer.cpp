//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainer.h"
#include "TreeTrainerImpl.h"
#include "BasePredictor.h"


TreeTrainer::TreeTrainer(CRefXXf inData, CRefXs strata) :
    impl_(createImpl_(inData, strata))
{
}

shared_ptr<TreeTrainerImplBase> TreeTrainer::createImpl_(CRefXXf inData, CRefXs strata)
{
    const size_t sampleCount = static_cast<size_t>(inData.rows());
    if (sampleCount < 0x100)
        return std::make_shared<TreeTrainerImpl<uint8_t>>(inData, strata);
    else if (sampleCount < 0x10000)
        return std::make_shared<TreeTrainerImpl<uint16_t>>(inData, strata);
    else if (sampleCount < 0x100000000)
        return std::make_shared<TreeTrainerImpl<uint32_t>>(inData, strata);
    else
        return std::make_shared<TreeTrainerImpl<uint64_t>>(inData, strata);
}

unique_ptr<BasePredictor> TreeTrainer::train(CRefXd outData, CRefXd weights, const StumpOptions& options) const
{
    PROFILE::PUSH(PROFILE::STUMP_TRAIN);
    unique_ptr<BasePredictor> pred =  impl_->train(outData, weights, options);
    PROFILE::POP();
    return pred;
}

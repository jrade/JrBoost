//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainer.h"

#include "BasePredictor.h"
#include "TreeTrainerImpl.h"


TreeTrainer::TreeTrainer(CRefXXfc inData, CRefXs strata) :
    impl_(createImpl_(inData, strata))
{
}

TreeTrainer::~TreeTrainer() = default;


unique_ptr<TreeTrainerImplBase> TreeTrainer::createImpl_(CRefXXfc inData, CRefXs strata)
{
    const size_t sampleCount = static_cast<size_t>(inData.rows());
    if (sampleCount < 0x100)
        return std::make_unique<TreeTrainerImpl<uint8_t>>(inData, strata);
    else if (sampleCount < 0x10000)
        return std::make_unique<TreeTrainerImpl<uint16_t>>(inData, strata);
    else if (sampleCount < 0x100000000)
        return std::make_unique<TreeTrainerImpl<uint32_t>>(inData, strata);
    else
        return std::make_unique<TreeTrainerImpl<uint64_t>>(inData, strata);
}

unique_ptr<BasePredictor> TreeTrainer::train(
    CRefXd outData, CRefXd weights, const TreeOptions& options, size_t threadCount) const
{
    PROFILE::PUSH(PROFILE::TREE_TRAIN);
    unique_ptr<BasePredictor> pred = impl_->train(outData, weights, options, threadCount);
    PROFILE::POP();
    return pred;
}

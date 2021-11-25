//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "ForestTrainer.h"

#include "BaseOptions.h"
#include "BasePredictor.h"
#include "TreeTrainer.h"


ForestTrainer::ForestTrainer(CRefXXfc inData, CRefXu8 strata) :
    treeTrainer_(createTreeTrainer_(inData, strata))
{
}

ForestTrainer::~ForestTrainer() = default;

unique_ptr<const TreeTrainerBase>  ForestTrainer::createTreeTrainer_(CRefXXfc inData, CRefXu8 strata)
{
    const size_t sampleCount = static_cast<size_t>(inData.rows());
    if (sampleCount <= 1 << 8)
        return std::make_unique<TreeTrainer<uint8_t>>(inData, strata);
    else if (sampleCount <= 1 << 16)
        return std::make_unique<TreeTrainer<uint16_t>>(inData, strata);
    else if (sampleCount <= 1LL << 32)
        return std::make_unique<TreeTrainer<uint32_t>>(inData, strata);
    else
        return std::make_unique<TreeTrainer<uint64_t>>(inData, strata);
}

unique_ptr<const BasePredictor> ForestTrainer::train(
    CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const
{
    size_t forestSize = options.forestSize();
    if (forestSize == 1)
        return treeTrainer_->train(outData, weights, options, threadCount);

    vector<unique_ptr<const BasePredictor>> basePredictors(forestSize);
    for (size_t k = 0; k != forestSize; ++k)
        basePredictors[k] = treeTrainer_->train(outData, weights, options, threadCount);
    return ForestPredictor::createInstance(move(basePredictors));
}

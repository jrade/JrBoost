//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "TreeTrainer.h"

#include "BaseOptions.h"
#include "BasePredictor.h"
#include "TreeTrainerImpl.h"


unique_ptr<TreeTrainer> TreeTrainer::createInstance(CRefXXfc inData, CRefXu8 strata)
{
    const size_t sampleCount = static_cast<size_t>(inData.rows());
    if (sampleCount <= 1 << 8)
        return std::make_unique<TreeTrainerImpl<uint8_t>>(inData, strata);
    else if (sampleCount <= 1 << 16)
        return std::make_unique<TreeTrainerImpl<uint16_t>>(inData, strata);
    else if (sampleCount <= 1LL << 32)
        return std::make_unique<TreeTrainerImpl<uint32_t>>(inData, strata);
    else
        return std::make_unique<TreeTrainerImpl<uint64_t>>(inData, strata);
}


unique_ptr<BasePredictor>
TreeTrainer::train(CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const
{
    const size_t forestSize = options.forestSize();
    if (forestSize == 1)
        return trainImpl0_(outData, weights, options, threadCount);

    vector<unique_ptr<BasePredictor>> basePredictors(forestSize);
    for (size_t k = 0; k != forestSize; ++k)
        basePredictors[k] = trainImpl0_(outData, weights, options, threadCount);
    return ForestPredictor::createInstance(move(basePredictors));
}

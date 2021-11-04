//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "ForestOptions.h"
#include "ForestPredictor.h"
#include "TreeTrainer.h"


class ForestTrainer
{
public:
    ForestTrainer(CRefXXfc inData, CRefXu8 strata) : treeTrainer_(inData, strata) {}
    ~ForestTrainer() = default;

    unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const ForestOptions& options, size_t threadCount) const
    {
        size_t forestSize = options.forestSize();
        if (forestSize == 1)
            return treeTrainer_.train(outData, weights, options, threadCount);

        vector<unique_ptr<BasePredictor>> basePredictors(forestSize);
        for (size_t k = 0; k < forestSize; ++k)
            basePredictors[k] = treeTrainer_.train(outData, weights, options, threadCount);
        return std::make_unique<ForestPredictor>(move(basePredictors));
    }

// deleted:
    ForestTrainer(const TreeTrainer&) = delete;
    ForestTrainer& operator=(const TreeTrainer&) = delete;

private:
    TreeTrainer treeTrainer_;
};

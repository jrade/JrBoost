//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class BaseOptions;
class BasePredictor;
class TreeTrainerBase;


class ForestTrainer {
public:
    ForestTrainer(CRefXXfc inData, CRefXu8 strata);
    ~ForestTrainer();

    unique_ptr<const BasePredictor>
    train(CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const;

    // deleted:
    ForestTrainer(const ForestTrainer&) = delete;
    ForestTrainer& operator=(const ForestTrainer&) = delete;

private:
    static unique_ptr<const TreeTrainerBase> createTreeTrainer_(CRefXXfc inData, CRefXu8 strata);

    const unique_ptr<const TreeTrainerBase> treeTrainer_;
};

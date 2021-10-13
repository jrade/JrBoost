//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class TreeOptions;
class TreeTrainerImplBase;
class BasePredictor;


class TreeTrainer
{
public:
    TreeTrainer(CRefXXfc inData, CRefXs strata);
    ~TreeTrainer();

    unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const TreeOptions& options, size_t threadCount) const;

// deleted:
    TreeTrainer(const TreeTrainer&) = delete;
    TreeTrainer& operator=(const TreeTrainer&) = delete;

private:
    static unique_ptr<TreeTrainerImplBase> createImpl_(CRefXXfc inData, CRefXs strata);

    const unique_ptr<const TreeTrainerImplBase> impl_;
};

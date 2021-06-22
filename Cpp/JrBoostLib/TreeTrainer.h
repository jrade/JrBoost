//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class StumpOptions;
class TreeTrainerImplBase;
class BasePredictor;


class TreeTrainer
{
public:
    TreeTrainer(CRefXXf inData, CRefXs strata);
    ~TreeTrainer() = default;

    unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const;

// deleted:
    TreeTrainer(const TreeTrainer&) = delete;
    TreeTrainer& operator=(const TreeTrainer&) = delete;

private:
    static shared_ptr<TreeTrainerImplBase> createImpl_(CRefXXf inData, CRefXs strata);

    const shared_ptr<const TreeTrainerImplBase> impl_;
};

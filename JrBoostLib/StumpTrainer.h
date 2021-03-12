//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class StumpOptions;
class StumpTrainerImplBase;
class SimplePredictor;


class StumpTrainer
{
public:
    StumpTrainer(CRefXXf inData, CRefXs strata);
    ~StumpTrainer() = default;

    unique_ptr<SimplePredictor> train(CRefXd outData, CRefXd weights, const StumpOptions& options) const;

// deleted:
    StumpTrainer(const StumpTrainer&) = delete;
    StumpTrainer& operator=(const StumpTrainer&) = delete;

private:
    static shared_ptr<StumpTrainerImplBase> createImpl_(CRefXXf inData, CRefXs strata);

    const shared_ptr<const StumpTrainerImplBase> impl_;
};

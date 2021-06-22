//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#define USE_TREE_TRAINER 1

class BoostOptions;
class BoostPredictor;

#if USE_TREE_TRAINER
class TreeTrainer;
using BaseTrainer = TreeTrainer;
#else
class StumpTrainer;
using BaseTrainer = StumpTrainer;
#endif


class BoostTrainer {
public:
    BoostTrainer(ArrayXXf inData, ArrayXs outData, optional<ArrayXd> weights);
    ~BoostTrainer() = default;

    shared_ptr<BoostPredictor> train(const BoostOptions& opt) const;

 // deleted:
    BoostTrainer(const BoostTrainer&) = delete;
    BoostTrainer& operator=(const  BoostTrainer&) = delete;

private:
    double calculateLor0_() const;
    shared_ptr<BoostPredictor> trainAda_(const BoostOptions& opt) const;
    shared_ptr<BoostPredictor> trainLogit_(const BoostOptions& opt) const;
    shared_ptr<BoostPredictor> trainRegularizedLogit_(const BoostOptions& opt) const;
    static void overflow_ [[noreturn]] (const BoostOptions& opt);

private:
    const ArrayXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const ArrayXs rawOutData_;
    const ArrayXd outData_;
    const optional<ArrayXd> weights_;
    const shared_ptr<BaseTrainer> baseTrainer_;
    const double lor0_;
};

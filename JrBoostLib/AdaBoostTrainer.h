#pragma once

#include "BoostOptions.h"
#include "StumpTrainer.h"

class BoostPredictor;


class AdaBoostTrainer {
public:
    AdaBoostTrainer(CRefXXf inData, const ArrayXd& outData, const ArrayXd& weights);
    ~AdaBoostTrainer() = default;
    BoostPredictor train(const BoostOptions& opt) const;

 // deleted:
    AdaBoostTrainer() = delete;
    AdaBoostTrainer(const AdaBoostTrainer&) = delete;
    AdaBoostTrainer& operator=(const  AdaBoostTrainer&) = delete;

private:
    CRefXXf inData_;
    ArrayXd outData_;
    ArrayXd weights_;
    StumpTrainer baseTrainer_;
};

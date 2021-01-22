#pragma once

#include "BoostOptions.h"
#include "StumpTrainer.h"

class BoostPredictor;


class AdaBoostTrainer {
public:
    AdaBoostTrainer(ArrayXXf inData, ArrayXs outData, ArrayXd weights);  // stores copies the arrays
    ~AdaBoostTrainer() = default;
    BoostPredictor train(const BoostOptions& opt) const;

 // deleted:
    AdaBoostTrainer() = delete;
    AdaBoostTrainer(const AdaBoostTrainer&) = delete;
    AdaBoostTrainer& operator=(const  AdaBoostTrainer&) = delete;

private:
    ArrayXXf inData_;
    ArrayXs rawOutData_;
    ArrayXd outData_;
    ArrayXd weights_;
    StumpTrainer baseTrainer_;
};

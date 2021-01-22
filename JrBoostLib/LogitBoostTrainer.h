#pragma once

#include "BoostOptions.h"
#include "StumpTrainer.h"

class BoostPredictor;


class LogitBoostTrainer {
public:
    LogitBoostTrainer(ArrayXXf inData, ArrayXs outData, ArrayXd weights);  // stores copies the arrays
    ~LogitBoostTrainer() = default;
    BoostPredictor train(const BoostOptions& opt) const;

// deleted:
    LogitBoostTrainer() = delete;
    LogitBoostTrainer(const LogitBoostTrainer&) = delete;
    LogitBoostTrainer& operator=(const  LogitBoostTrainer&) = delete;

private:
    ArrayXXf inData_;
    ArrayXs rawOutData_;
    ArrayXd outData_;
    ArrayXd weights_;
    StumpTrainer baseTrainer_;
};

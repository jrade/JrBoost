#pragma once

#include "BoostOptions.h"
#include "StumpTrainer.h"

class BoostPredictor;


class LogitBoostTrainer {
public:
    LogitBoostTrainer(CRefXXf inData, const ArrayXd& outData, const ArrayXd& weights);
    ~LogitBoostTrainer() = default;
    BoostPredictor train(const BoostOptions& opt) const;

    // deleted:
    LogitBoostTrainer() = delete;
    LogitBoostTrainer(const LogitBoostTrainer&) = delete;
    LogitBoostTrainer& operator=(const  LogitBoostTrainer&) = delete;

private:
    CRefXXf inData_;
    ArrayXd outData_;
    ArrayXd weights_;
    StumpTrainer baseTrainer_;
};

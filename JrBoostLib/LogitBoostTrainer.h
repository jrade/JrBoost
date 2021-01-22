#pragma once

#include "StumpTrainer.h"

class BoostOptions;
class BoostPredictor;


class LogitBoostTrainer {
public:
    LogitBoostTrainer(CRefXXf inData, ArrayXs outData);
    ~LogitBoostTrainer() = default;
    BoostPredictor train(const BoostOptions& opt) const;
    Eigen::ArrayXXd trainAndPredict(ArrayXXf testInData, const vector<BoostOptions>& opt) const;

    // deleted:
    LogitBoostTrainer() = delete;
    LogitBoostTrainer(const LogitBoostTrainer&) = delete;
    LogitBoostTrainer& operator=(const  LogitBoostTrainer&) = delete;

private:
    const CRefXXf inData_;
    ArrayXs rawOutData_;                    // pesky const issue - can not iterate over const array
    const ArrayXd outData_;
    const StumpTrainer baseTrainer_;
};

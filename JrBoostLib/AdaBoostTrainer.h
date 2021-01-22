#pragma once

#include "StumpTrainer.h"

class BoostOptions;
class BoostPredictor;


class AdaBoostTrainer {
public:
    AdaBoostTrainer(CRefXXf inData, ArrayXs outData);
    ~AdaBoostTrainer() = default;
    BoostPredictor train(const BoostOptions& opt) const;
    Eigen::ArrayXXd trainAndPredict(ArrayXXf testInData, const vector<BoostOptions>& opt) const;

 // deleted:
    AdaBoostTrainer() = delete;
    AdaBoostTrainer(const AdaBoostTrainer&) = delete;
    AdaBoostTrainer& operator=(const  AdaBoostTrainer&) = delete;

private:
    const CRefXXf inData_;
    ArrayXs rawOutData_;                    // pesky const issue - can not iterate over const array
    const ArrayXd outData_;
    const StumpTrainer baseTrainer_;
};

#pragma once

#include "StumpTrainer.h"

class BoostOptions;
class AbstractPredictor;


class BoostTrainer {
public:
    BoostTrainer(CRefXXf inData, RefXs outData);
    ~BoostTrainer() = default;

    unique_ptr<AbstractPredictor> train(const BoostOptions& opt) const;
    ArrayXd trainAndEval(CRefXXf testInData, CRefXs testOutData, const vector<BoostOptions>& opt) const;

 // deleted:
    BoostTrainer() = delete;
    BoostTrainer(const BoostTrainer&) = delete;
    BoostTrainer& operator=(const  BoostTrainer&) = delete;

    inline static int threadCount = std::thread::hardware_concurrency();

private:
    unique_ptr<AbstractPredictor> trainAda_(const BoostOptions& opt) const;
    unique_ptr<AbstractPredictor> trainLogit_(const BoostOptions& opt) const;

private:
    const CRefXXf inData_;
    RefXs rawOutData_;                    // pesky const issue - can not iterate over const array
    const ArrayXd outData_;
    const StumpTrainer baseTrainer_;
};

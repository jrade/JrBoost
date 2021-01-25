#pragma once

#include "StumpTrainer.h"

class BoostOptions;
class BoostPredictor;


class BoostTrainer {
public:
    BoostTrainer(CRefXXf inData, ArrayXs outData);
    ~BoostTrainer() = default;

    BoostPredictor train(const BoostOptions& opt) const;
    ArrayXd trainAndEval(CRefXXf testInData, ArrayXs testOutData, const vector<BoostOptions>& opt) const;

 // deleted:
    BoostTrainer() = delete;
    BoostTrainer(const BoostTrainer&) = delete;
    BoostTrainer& operator=(const  BoostTrainer&) = delete;

private:
    BoostPredictor trainAda_(const BoostOptions& opt) const;
    BoostPredictor trainLogit_(const BoostOptions& opt) const;

    const CRefXXf inData_;
    ArrayXs rawOutData_;                    // pesky const issue - can not iterate over const array
    const ArrayXd outData_;
    const StumpTrainer baseTrainer_;
};

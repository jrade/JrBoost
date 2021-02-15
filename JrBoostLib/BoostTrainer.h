#pragma once

class BoostOptions;
class AbstractPredictor;
class StumpTrainer;


class BoostTrainer {
public:
    BoostTrainer(ArrayXXf inData, ArrayXs outData);
    ~BoostTrainer() = default;

    unique_ptr<AbstractPredictor> train(const BoostOptions& opt) const;
    ArrayXd trainAndEval(CRefXXf testInData, CRefXs testOutData, const vector<BoostOptions>& opt) const;

 // deleted:
    BoostTrainer(const BoostTrainer&) = delete;
    BoostTrainer& operator=(const  BoostTrainer&) = delete;

private:
    unique_ptr<AbstractPredictor> trainAda_(const BoostOptions& opt) const;
    unique_ptr<AbstractPredictor> trainLogit_(const BoostOptions& opt) const;

private:
    const ArrayXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const ArrayXs rawOutData_;
    const ArrayXd outData_;
    const shared_ptr<StumpTrainer> baseTrainer_;
    const double f0_;

    inline static thread_local ArrayXd F_;
    inline static thread_local ArrayXd adjWeights_;
};

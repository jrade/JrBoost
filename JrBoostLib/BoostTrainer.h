#pragma once

class BoostOptions;
class BoostPredictor;
class StumpTrainer;


class BoostTrainer {
public:
    BoostTrainer(ArrayXXf inData, ArrayXs outData, optional<ArrayXd> weights);
    ~BoostTrainer() = default;

    unique_ptr<BoostPredictor> train(const BoostOptions& opt) const;

    ArrayXd trainAndEval(
        CRefXXf testInData,
        CRefXs testOutData,
        const vector<BoostOptions>& opt,
        function<tuple<double, double, double>(CRefXs, CRefXd)> lossFun
    ) const;

 // deleted:
    BoostTrainer(const BoostTrainer&) = delete;
    BoostTrainer& operator=(const  BoostTrainer&) = delete;

private:
    double calculateF0_() const;
    unique_ptr<BoostPredictor> trainAda_(const BoostOptions& opt) const;
    unique_ptr<BoostPredictor> trainLogit_(const BoostOptions& opt) const;

private:
    const ArrayXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const ArrayXs rawOutData_;
    const ArrayXd outData_;
    const optional<ArrayXd> weights_;
    const shared_ptr<StumpTrainer> baseTrainer_;
    const double f0_;

    inline static thread_local ArrayXd F_;
    inline static thread_local ArrayXd adjWeights_;
};

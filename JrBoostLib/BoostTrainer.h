//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

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
        function<Array3d(CRefXs, CRefXd)> lossFun
    ) const;

 // deleted:
    BoostTrainer(const BoostTrainer&) = delete;
    BoostTrainer& operator=(const  BoostTrainer&) = delete;

private:
    double calculateLor0_() const;
    unique_ptr<BoostPredictor> trainAda_(BoostOptions opt) const;
    unique_ptr<BoostPredictor> trainAlpha_(BoostOptions opt) const;

private:
    const ArrayXXf inData_;
    const size_t sampleCount_;
    const size_t variableCount_;
    const ArrayXs rawOutData_;
    const ArrayXd outData_;
    const optional<ArrayXd> weights_;
    const shared_ptr<StumpTrainer> baseTrainer_;
    const double lor0_;
};

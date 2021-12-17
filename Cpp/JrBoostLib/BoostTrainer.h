//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class BoostOptions;
class Predictor;
class TreeTrainer;


class BoostTrainer {   // immutable class
public:
    BoostTrainer(
        ArrayXXfc inData, ArrayXu8 outData, optional<ArrayXd> weights = std::nullopt,
        optional<ArrayXu8> strata = std::nullopt);
    BoostTrainer(const BoostTrainer&) = delete;
    BoostTrainer& operator=(const BoostTrainer&) = delete;
    ~BoostTrainer();

    shared_ptr<Predictor> train(const BoostOptions& opt, size_t threadCount = 0) const;


private:
    static void validateData_(CRefXXfc inData, CRefXu8 outData, optional<CRefXd> weights, optional<CRefXu8> strata);
    double getGlobalLogOddsRatio_() const;

    shared_ptr<Predictor> trainAda_(const BoostOptions& opt, size_t threadCount) const;
    shared_ptr<Predictor> trainLogit_(const BoostOptions& opt, size_t threadCount) const;
    shared_ptr<Predictor> trainRegularizedLogit_(const BoostOptions& opt, size_t threadCount) const;
    static void overflow_ [[noreturn]] (const BoostOptions& opt);

private:
    const size_t sampleCount_;
    const size_t variableCount_;
    const ArrayXXfc inData_;
    const ArrayXd outData_;
    const optional<ArrayXd> weights_;
    const ArrayXu8 strata_;
    const double globaLogOddsRatio_;
    const unique_ptr<TreeTrainer> treeTrainer_;
};

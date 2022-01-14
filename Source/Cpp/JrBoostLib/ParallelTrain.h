//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class BoostOptions;
class BoostTrainer;
class Predictor;


vector<shared_ptr<Predictor>> parallelTrain(const BoostTrainer& trainer, const vector<BoostOptions>& opt);

ArrayXXdc parallelTrainAndPredict(const BoostTrainer& trainer, const vector<BoostOptions>& opt, CRefXXfc testInData);

ArrayXd parallelTrainAndEval(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    function<double(CRefXu8, CRefXd, optional<CRefXd>)> lossFun, CRefXXfc testInData, CRefXu8 testOutData,
    optional<CRefXd> testWeights = std::nullopt);

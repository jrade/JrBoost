//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class BoostTrainer;
class BoostOptions;
class BoostPredictor;


vector<shared_ptr<BoostPredictor>> parallelTrain(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt);

ArrayXXdc parallelTrainAndPredict(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    CRefXXfc testInData
);

ArrayXd parallelTrainAndEval(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    CRefXXfc testInData, CRefXs testOutData, function<double(CRefXs, CRefXd)> lossFun
);

ArrayXd parallelTrainAndEvalWeighted(
    const BoostTrainer& trainer, const vector<BoostOptions>& opt,
    CRefXXfc testInData, CRefXs testOutData, CRefXd testWeights, function<double(CRefXs, CRefXd, CRefXd)> lossFun
);

#  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import random
import numpy as np
import jrboost


#-----------------------------------------------------------------------------------------------------------------------

def train(inData, outData, param, *, samples = None, weights = None, strata = None):

    repCount = param.get('repCount', 1)
    minimizeAlgorithm = param['minimizeAlgorithm']
    boostParamGrid = param['boostParamGrid']
    minimizeParam = param.get('minimizeParam', {})

    bestBoostParams = []
    for _ in range(repCount):
        bestBoostParams += minimizeAlgorithm(
            lambda boostParams: _trainAndEval(boostParams, inData, outData, param, samples, weights, strata),
            boostParamGrid,
            minimizeParam)

    if samples is not None:
        inData = inData[samples, :]
        outData = outData[samples]
        weights = None if weights is None else weights[samples]
        strata = None if strata is None else strata[samples]

    trainer = jrboost.BoostTrainer(inData, outData, weights = weights, strata = strata)
    pred = jrboost.Predictor.createEnsemble([trainer.train(bp) for bp in bestBoostParams])

    return pred, _medianBoostParam(bestBoostParams)


def _trainAndEval(boostParams, inData, outData, param, samples, weights, strata):

    foldCount = param['foldCount']
    targetLossFun = param['targetLossFun']
    boostParamGrid = param['boostParamGrid']

    loss = np.zeros((len(boostParams),))

    for trainSamples, testSamples in jrboost.stratifiedRandomFolds(outData if strata is None else strata, foldCount, samples):


        trainInData = inData[trainSamples, :]
        trainOutData = outData[trainSamples]
        trainWeights = None if weights is None else weights[trainSamples]
        trainStrata = None if strata is None else strata[trainSamples]

        trainer = jrboost.BoostTrainer(trainInData, trainOutData, strata = trainStrata, weights = trainWeights)

        testInData = inData[testSamples, :]
        testOutData = outData[testSamples]
        testWeights = None if weights is None else weights[testSamples]

        if weights is None:
            loss += jrboost.parallelTrainAndEval(trainer, boostParams, testInData, testOutData, targetLossFun)
        else:
            loss += jrboost.parallelTrainAndEvalWeighted(
                trainer, boostParams, testInData, testOutData, testWeights, targetLossFun)

    return loss



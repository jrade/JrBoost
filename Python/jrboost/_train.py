#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import copy, random
import numpy as np
import jrboost


def minimizeGrid(f, grid, param = {}):

    bestCount = param.get('bestCount', 1)

    xList = [dict()]
    for key, values in grid.items():
        xList1 = []
        for x in xList:
            for value in values:
                x[key] = value
                xList1.append(copy.copy(x))
        xList = xList1

    yList = f(xList)
    bestIndices = list(np.argsort(yList)) 
    xList = [xList[i] for i in bestIndices[:bestCount]]

    return xList


def minimizePopulation(f, grid, param):

    cycleCount = param['cycleCount']
    populationCount = param['populationCount']
    survivorCount = param['survivorCount']
    bestCount = param['bestCount']

    xDict = copy.deepcopy(grid)
    k = 0
    while True:

        xDict = _sample(xDict, populationCount)
        xList = _split(xDict)
        yList = f(xList)
        bestIndices = list(np.argsort(yList))

        k += 1
        if k == cycleCount:
            xList = [xList[i] for i in bestIndices[:bestCount]]
            return xList

        xList = [xList[i] for i in bestIndices[:survivorCount]]
        xDict = _merge(xList)


def _divideRoundUp(n, m):
    return (n + m - 1) // m

# list a -> list b
# b is random sample from a, repeated if necessary
def _randomSample(a, n):
    k = _divideRoundUp(n, len(a))
    return random.sample(k * a, n)

# dict of lists a -> dict of lists b
# b[key] is a random sample of a[key]
def _sample(a, n):
    return {key : _randomSample(value, n) for key, value in a.items()}

# dict of lists a -> list of dicts b
# b[i][k] = a[k][i]
# the list a[k] must have the same length for all k
def _split(a):
    for value in a.values(): break
    n = len(value)
    return  [{key: value[i] for key, value in a.items()} for i in range(n)]

# list of dicts a -> dict of lists b
# b[k][i] = a[i][k]
# the dict a[i] must have the same keys for all i
def _merge(a):
    keys = a[0].keys()
    return {key : [item[key] for item in a] for key in keys}

#-----------------------------------------------------------------------------------------------------------------------

class TrivialPreprocessor:

    def __init__(self, inData, outData, variableCount = None):
        pass

    def __call__(self, inData):
        return inData

    def __bool__(self):
        return False


class TTestPreprocessor:

    def __init__(self, inData, outData, variableCount = None):
        self._variables = jrboost.tTestRank(inData, outData)
        if variableCount is not None:
            self._variables = self._variables[: variableCount]

    def __call__(self, inData):
        return inData[:, self._variables]

    def __bool__(self):
        return True

#-----------------------------------------------------------------------------------------------------------------------

def train(inData, outData, param, *, weights = None, strata = None):

    repCount = param.get('repCount', 1)
    minimizeAlgorithm = param['minimizeAlgorithm']
    boostParamGrid = param['boostParamGrid']
    minimizeParam = param.get('minimizeParam', {})
    PP = param.get('preprocessor', TrivialPreprocessor)

    bestBoostParams = []
    for _ in range(repCount):
        bestBoostParams += minimizeAlgorithm(
            lambda boostParams: _trainAndEval(boostParams, inData, outData, param, weights, strata),
            boostParamGrid,
            minimizeParam
        )

    if 'topVariableCount' in boostParamGrid:
        maxTopVariableCount = max(bp['topVariableCount'] for bp in bestBoostParams)
        pp = PP(inData, outData, maxTopVariableCount)
    else:
        pp = PP(inData, outData)

    inData = pp(inData)
    trainer = jrboost.BoostTrainer(inData, outData, weights = weights, strata = strata)
    pred = jrboost.Predictor.createEnsemble([trainer.train(bp) for bp in bestBoostParams])

    return pp, pred, _medianBoostParam(bestBoostParams)


def _trainAndEval(boostParams, inData, outData, param, weights, strata):

    PP = param.get('preprocessor', TrivialPreprocessor)
    foldCount = param['foldCount']
    targetLossFun = param['targetLossFun']
    boostParamGrid = param['boostParamGrid']

    loss = np.zeros((len(boostParams),))

    for trainSamples, testSamples in jrboost.stratifiedRandomFolds(outData if strata is None else strata, foldCount):

        trainInData = inData[trainSamples, :]
        trainOutData = outData[trainSamples]
        trainWeights = None if weights is None else weights[trainSamples]
        trainStrata = None if strata is None else strata[trainSamples]

        if 'topVariableCount' in boostParamGrid:
            maxTopVariableCount = max(bp['topVariableCount'] for bp in boostParams)
            pp = PP(trainInData, trainOutData, maxTopVariableCount)
        else:
            pp = PP(trainInData, trainOutData)
        trainInData = pp(trainInData)

        trainer = jrboost.BoostTrainer(trainInData, trainOutData, strata = trainStrata)

        testInData = pp(inData[testSamples, :])
        testOutData = outData[testSamples]
        if weights is None:
            loss += jrboost.parallelTrainAndEval(trainer, boostParams, testInData, testOutData, targetLossFun)
        else:
            testWeights = weights[testSamples]
            loss += jrboost.parallelTrainAndEvalWeighted(
                trainer, boostParams, testInData, testOutData, testWeights, targetLossFun)

    return loss


def _medianBoostParam(boostParams):
    keys = boostParams[0].keys()
    medianBp = { key : _roughMedian([boostParam[key] for boostParam in boostParams]) for key in keys}
    return medianBp

def _roughMedian(a):
    if len(a) % 2 == 0:
        return sorted(a)[len(a) // 2 - random.randint(0,1)]
    else:
        return sorted(a)[len(a) // 2]

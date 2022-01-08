#  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import copy, random, re
import numpy as np
import pandas as pd
import jrboost


def oneHotEncode(dataSeries):

    assert isinstance(dataSeries, pd.Series)

    labels = sorted(set([label.strip() for label in dataSeries]))
                    
    dataFrame = pd.DataFrame(
        index = dataSeries.index,
        columns = pd.Index(labels, name = dataSeries.name),
        data = 0
    )

    for sample in dataSeries.index:
        label = dataSeries[sample]
        dataFrame.loc[sample, label.strip()] = 1

    return dataFrame


def stratifiedRandomFolds(strata, foldCount, samples = None):

    assert foldCount >= 2

    if samples is None:
        tmp = list(enumerate(strata))
    else:
        tmp = [(i, strata[i]) for i in samples]
    random.shuffle(tmp)
    tmp.sort(key = lambda x: x[1])

    folds = [([], []) for _ in range(foldCount)]
    for (j, (i, _)) in enumerate(tmp):
        for foldIndex in range(foldCount):
            folds[foldIndex][foldIndex == (j % foldCount)].append(i)

    folds = [(sorted(trainSamples), sorted(testSamples)) for trainSamples, testSamples in folds]
    
    return folds


def stratifiedRandomSplit(strata, ratio, samples = None):

    assert 0.0 <= ratio <= 1.0

    if samples is None:
        tmp = list(enumerate(strata))
    else:
        tmp = [(i, strata[i]) for i in samples]
    random.shuffle(tmp)
    tmp.sort(key = lambda x: x[1])

    trainSamples = []
    testSamples = []
    theta = random.random()  # range [0.0, 1.0)
    for i, _ in tmp:
        theta += ratio
        if theta >= 1.0:
            trainSamples.append(i)
            theta -= 1.0
        else:
            testSamples.append(i)

    trainSamples = sorted(trainSamples)
    testSamples = sorted(testSamples)

    return trainSamples, testSamples

#-----------------------------------------------------------------------------------------------------------------------

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

        #if True:
        #    print(k)
        #    for key, value in grid.items():
        #        if len(value) == 1: continue
        #        value = xDict[key]
        #        def round(x): return int(1000000 * x + 0.5) / 1000000
        #        print(f'{key}: {round(np.quantile(value, 0.25))} / {round(np.quantile(value, 0.5))} / {round(np.quantile(value, 0.75))}')
        #    print()

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


def optimizeHyperParam(inData, outData, param, *, samples = None, weights = None, strata = None):

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

    return bestBoostParams


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

        loss += jrboost.parallelTrainAndEval(
            trainer, boostParams, targetLossFun, testInData, testOutData, weights = testWeights)

    return loss

#-----------------------------------------------------------------------------------------------------------------------

def medianBoostParam(boostParams):

    keys = boostParams[0].keys()
    medianBp = { key : _roughMedian([boostParam[key] for boostParam in boostParams]) for key in keys}
    return medianBp

def _roughMedian(a):

    if len(a) % 2 == 0:
        return sorted(a)[len(a) // 2 - random.randint(0,1)]
    else:
        return sorted(a)[len(a) // 2]

#-----------------------------------------------------------------------------------------------------------------------

_regCpp = re.compile('<built-in method (\S+) of PyCapsule object at 0x.{16}>')
_regPy = re.compile('<function (\S+) at 0x.{16}>')

def formatParam(obj, indent = 0):

    if isinstance(obj, dict):
        s = '{\n'
        for key, value in obj.items():
            s += (indent + 1) * '    ' + repr(key) + ': ' + formatParam(value, indent + 1) + '\n'
        s += indent * '    ' + '}'
        return s

    if isinstance(obj, list):
        return '[' + ', '.join(formatParam(a) for a in obj) + ']'

    if isinstance(obj, tuple):
        return '(' + ', '.join(formatParam(a) for a in obj) + ')'

    s = str(obj)

    m = _regCpp.match(s)
    if m: 
        return m.group(1)

    m = _regPy.match(s)
    if m: 
        return m.group(1)

    try:
        return obj.name
    except:
        return s

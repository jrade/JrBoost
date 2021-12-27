#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import copy, random
import numpy as np
import pandas as pd


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


#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

__all__ = ['oneHotEncode', 'stratifiedRandomFolds', 'stratifiedRandomSplit', 'minimizeGrid', 'minimizePopulation', 'findPath']

import copy, os, random, warnings
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


def stratifiedRandomFolds(outData, foldCount, samples = None):

    if samples is None:
        tmp = list(enumerate(outData))
    else:
        tmp = [(i, outData[i]) for i in samples]
    random.shuffle(tmp)
    tmp.sort(key = lambda x: x[1])

    folds = [([], []) for _ in range(foldCount)]
    for (j, (i, _)) in enumerate(tmp):
        for foldIndex in range(foldCount):
            folds[foldIndex][foldIndex == (j % foldCount)].append(i)

    folds = [(
        np.array(sorted(trainSamples)),
        np.array(sorted(testSamples))
    ) for trainSamples, testSamples in folds]
    
    return folds


def stratifiedRandomSplit(outData, ratio, samples = None):

    if samples is None:
        tmp = list(enumerate(outData))
    else:
        tmp = [(i, outData[i]) for i in samples]
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

    trainSamples = np.array(sorted(trainSamples))
    testSamples = np.array(sorted(testSamples))

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

#-----------------------------------------------------------------------------------------------------------------------

def minimizePopulation(f, grid, param):

    cycleCount = param['cycleCount']
    populationCount = param['populationCount']
    survivorCount = param['survivorCount']
    bestCount = param['bestCount']

    xDict = copy.deepcopy(grid)
    k = 0
    while True:

        xDict = _sampleDict(xDict, populationCount)
        xList = _split(xDict)
        yList = f(xList)
        bestIndices = list(np.argsort(yList))

        k += 1
        if k == cycleCount:
            xList = [xList[i] for i in bestIndices[:bestCount]]
            return xList

        xList = [xList[i] for i in bestIndices[:survivorCount]]
        xDict = _merge(xList)


# random sample of size n from the list a
def _sampleList(a, n):
    k = (n + len(a) - 1) // len(a)
    a = k * a
    assert len(a) >= n
    random.shuffle(a)
    return a[:n]

# dictionary with values that are lists -> dictionary with values that are random samples from those lists
def _sampleDict(a, n):
    return {key : _sampleList(valueList, n) for key, valueList in a.items()}


# dictionary with values thar are lists -> list of dictionaries
# all values of the input dictionary must be lists of he same length
def _split(a):
    n = len(next(iter(a.values())))
    return  [ {key: valueList[i] for key, valueList in a.items()} for i in range(n) ]

# list of dictionaries -> dictionary with values thar are lists
# all dictionaries in the list must have the same keys
def _merge(a):
    keys = a[0].keys()
    return {key : [b[key] for b in a] for key in keys}

#-----------------------------------------------------------------------------------------------------------------------

def findPath(filePath):
    adjFilePath = filePath
    i = 0
    while not os.path.exists(adjFilePath):
        if i == 10:
            raise RuntimeError(f'Cannot find the file {filePath}')
        adjFilePath = '../' + adjFilePath
        i += 1
    return adjFilePath

#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

__all__ = ['oneHotEncode', 'stratifiedRandomFolds', 'stratifiedRandomSplit', 'optimizeDynamic', 'optimizeGrid']

import copy, random, warnings
import numpy as np
import pandas as pd
import jrboost


def oneHotEncode(dataSeries, separator = None):

    assert isinstance(dataSeries, pd.Series)

    if separator is None:
        labels = sorted(set([label.strip() for label in dataSeries]))
    else:
        labels = sorted(set([label.strip() for s in dataSeries for label in s.split(separator) ]))
                    
    dataFrame = pd.DataFrame(
        index = dataSeries.index,
        columns = pd.Index(labels, name = dataSeries.name),
        data = 0
    )

    if separator is None:
        for sample in dataSeries.index:
            label = dataSeries[sample]
            dataFrame.loc[sample, label.strip()] = 1
    else:
        for sample in dataSeries.index:
            for label in dataSeries[sample].split(separator):
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
            testSampes.append(i)

    return np.array(sorted(trainSamples)), np.array(sorted(testSamples))

#-----------------------------------------------------------------------------------------------------------------------

def optimizeDynamic(cvParam, evalFun):

    cycleCount = cvParam['cycleCount']
    populationCount = cvParam['populationCount']
    survivorCount = cvParam['survivorCount']
    bpValues = cvParam['boostParamValues']

    bpValues = copy.deepcopy(bpValues)
    k = 0
    while True:

        for values in bpValues.values():
            valueCount = len(values)
            values *= (populationCount + valueCount - 1) // valueCount
            random.shuffle(values)
            del values[populationCount:]

        optionList = [jrboost.BoostOptions() for _ in range(populationCount)]
        for name, values in bpValues.items():
            for i in range(populationCount):
                setattr(optionList[i], name, values[i])

        loss = evalFun(optionList)
        sortedIndices = list(np.argsort(loss))

        k += 1
        if k == cycleCount:
            break

        del sortedIndices[survivorCount:]
        for values in bpValues.values():
            values[:] = [values[i] for i in sortedIndices]
            values.sort()

    # finalize

    bestOptionCount = cvParam['bestOptionCount']
    ultraBoost = cvParam.get('ultraBoost', None)
    bagSize = cvParam.get('bagSize', None)

    del sortedIndices[bestOptionCount:]
    optionList = [optionList[i] for i in sortedIndices]

    if ultraBoost is not None:
        for opt in optionList:
            opt.iterationCount *= ultraBoost
            opt.eta /= ultraBoost

    if bagSize is not None:
        optionList *= bagSize

    return optionList

#-----------------------------------------------------------------------------------------------------------------------

def optimizeGrid(cvParam, evalFun):

    bestOptionCount = cvParam['bestOptionCount']
    ultraBoost = cvParam.get('ultraBoost', None)
    bagSize = cvParam['bagSize']

    bpValues = cvParam['boostParamValues']

    optionList = [jrboost.BoostOptions()]
    for name, values in bpValues.items():
        tmp = []
        for opt in optionList:
            for value in values:
                setattr(opt, name, value)
                tmp.append(copy.copy(opt))
        optionList = tmp

    loss = evalFun(optionList)
    
    bestIndices = list(np.argsort(loss)) 
    del bestIndices[bestOptionCount:]
    bestOptionList = [optionList[i] for i in bestIndices]

    if ultraBoost is not None:
        for opt in bestOptionList:
            opt.iterationCount *= ultraBoost
            opt.eta /= ultraBoost

    if bagSize is not None:
        optionList *= bagSize

    return optionList

import random, warnings
import numpy as np
import pandas as pd


def oneHotEncode(dataSeries):

    assert isinstance(dataSeries, pd.Series)

    samples = dataSeries.index
    labels = sorted(set(dataSeries))
    columns = pd.Index(labels, name = dataSeries.name)                          # Simplify ??????
    dataFrame = pd.DataFrame(index = samples, columns = columns, data = 0)
    for sample in dataSeries.index:
        dataFrame.loc[sample, dataSeries[sample]] = 1

    return dataFrame


def stratifiedRandomFolds(outData, foldCount):

    tmp = list(enumerate(outData))
    random.shuffle(tmp)
    tmp.sort(key = lambda x: x[1])
    folds = [([], []) for _ in range(foldCount)]
    for (i, (sampleIndex, x)) in enumerate(tmp):
        for foldIndex in range(foldCount):
            folds[foldIndex][foldIndex == (i % foldCount)].append(sampleIndex)
    for trainSamples, testSamples in folds:
        trainSamples.sort()
        testSamples.sort()
    return folds


def linLoss(outData, predData):     # predData should conain logloss ratios

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning) 

        return np.sum(
            outData[:, np.newaxis] / (1.0 + np.exp(predData))
            + (1 - outData[:, np.newaxis]) / (1.0 + np.exp(-predData)),
            axis = 0
        )


def logLoss(outData, predData):     # predData should conain logloss ratios

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning) 

        logOneProb = (
            (predData >= 0) * (-np.log1p(np.exp(-predData)))
            + (predData < 0) * (predData - np.log1p(np.exp(predData)))
        )

        logZeroProb = (
            (predData >= 0) * (-predData - np.log1p(np.exp(-predData)))
            + (predData < 0) * (-np.log1p(np.exp(predData)))
        )

        return -np.sum(
            outData[:, np.newaxis] * logOneProb
            + (1 - outData[:, np.newaxis]) * logZeroProb,
            axis = 0
        )

def formatTime(t):
    h = int(t / 3600)
    t -= 3600 * h;
    m = int(t / 60)
    t -= 60 * m
    s = int(t)
    return f'{h}:{m:02}:{s:02}'

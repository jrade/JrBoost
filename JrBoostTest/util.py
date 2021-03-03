import os, random, warnings
import numpy as np
import pandas as pd
import jrboost


# convert log odds ratio to probability
def lorToP(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 1.0 / (1.0 + np.exp(-a))


def oneHotEncode(dataSeries):

    assert isinstance(dataSeries, pd.Series)

    samples = dataSeries.index
    labels = sorted(set([label for s in dataSeries for label in s.split(';') ]))
    #labels = sorted(set(dataSeries))
                    
    columns = pd.Index(labels, name = dataSeries.name)
    dataFrame = pd.DataFrame(index = samples, columns = columns, data = 0)
    for sample in dataSeries.index:
        for label in dataSeries[sample].split(';'):
            dataFrame.loc[sample, label] = 1

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


def findPath(path):

    i = 0
    while True:
        if os.path.exists(path):
            return path
        if (i >= 10):
            raise RuntimeError(f'Unable to find {path}')
        path = '../' + path
        i += 1


def formatTime(t):
    h = int(t / 3600)
    t -= 3600 * h;
    m = int(t / 60)
    t -= 60 * m
    s = int(t)
    return f'{h}:{m:02}:{s:02}'

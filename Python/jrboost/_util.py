#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import random
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

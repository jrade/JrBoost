import os, random, warnings
import numpy as np
import pandas as pd
import jrboost


def oneHotEncode(dataSeries):

    assert isinstance(dataSeries, pd.Series)

    samples = dataSeries.index
    labels = sorted(set([label for s in dataSeries for label in s.split(';') ]))
    #labels = sorted(set(dataSeries))
                    
    columns = pd.Index(labels, name = dataSeries.name)                          # Simplify ??????
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

def linLoss(outData, predData):     # predData should contain logloss ratios

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning) 

        falseNeg = np.sum(outData / (1.0 + np.exp(predData)))
        falsePos = np.sum((1 - outData) / (1.0 + np.exp(-predData)))
        loss = (falsePos, falseNeg, falsePos + falseNeg)

        return loss


#def logLoss(outData, predData):     # predData should conain logloss ratios

#    with warnings.catch_warnings():
#        warnings.filterwarnings("ignore", category=RuntimeWarning) 

#        logOneProb = (
#            (predData >= 0) * (-np.log1p(np.exp(-predData)))
#            + (predData < 0) * (predData - np.log1p(np.exp(predData)))
#        )

#        logZeroProb = (
#            (predData >= 0) * (-predData - np.log1p(np.exp(-predData)))
#            + (predData < 0) * (-np.log1p(np.exp(predData)))
#        )

#        return -np.sum(
#            outData[:, np.newaxis] * logOneProb
#            + (1 - outData[:, np.newaxis]) * logZeroProb,
#            axis = 0
#        )


def trainAndEvalInternal(inData, outData, trainSamples, testSamples, optionsList, lossFun, rankFun = None):

    maxVariableCount = max(opt.base.topVariableCount for opt in optionsList)
    if rankFun is None:
        trainInData = inData[trainSamples, :maxVariableCount]
        testInData = inData[testSamples, :maxVariableCount]
    else:
        rankedVariables = rankFun(inData, outData, trainSamples)[:maxVariableCount]
        trainInData = inData[trainSamples[:, np.newaxis], rankedVariables]
        testInData = inData[testSamples[:, np.newaxis], rankedVariables]

    trainOutData = outData[trainSamples]
    testOutData = outData[testSamples]

    trainer = jrboost.BoostTrainer(trainInData, trainOutData)
    scores =  trainer.trainAndEval(testInData, testOutData, optionsList, lossFun);
    return scores


#does not take average of several opts, fix that
def trainAndPredictInternal(inData, outData, trainSamples, testSamples, opt, rankFun = None):

    opt = opt[0]

    try:
        maxVariableCount = opt.base.topVariableCount
    except:
        maxVariableCount = max(opt1.base.topVariableCount for opt1 in opt)

    if rankFun is None:
        trainInData = inData[trainSamples, :maxVariableCount]
        testInData = inData[testSamples, :maxVariableCount]
    else:
        rankedVariables = rankFun(inData, outData, trainSamples)[:maxVariableCount]
        trainInData = inData[trainSamples[:, np.newaxis], rankedVariables]
        testInData = inData[testSamples[:, np.newaxis], rankedVariables]

    trainOutData = outData[trainSamples]
    testOutData = outData[testSamples]

    trainer = jrboost.BoostTrainer(trainInData, trainOutData)

    try:
        predictor = trainer.train(opt);
        predOutData = predictor.predict(testInData);
    except Exception:
        predOutDataList = []
        for opt1 in opt:
            predictor = trainer.train(opt1);
            predOutData = predictor.predict(testInData);
            predOutDataList.append(predOutData)
        predOutData = np.median(np.array(predOutDataList), axis = 0)

    return predOutData

# takes average of several opts
# no support for rankFun - could be fixed
def trainAndPredictExternal(trainInData, trainOutData, testInData, opts):

    testSampleCount = testInData.shape[0]
    testOutData = np.zeros((testSampleCount,))

    trainer = jrboost.BoostTrainer(trainInData, trainOutData)
    for opt in opts:
        predictor = trainer.train(opt)
        testOutData += predictor.predict(testInData)
    testOutData /= len(opts)
    return testOutData


def findPath(path):

    i = 0
    while True:
        if os.path.exists(path):
            return path
        if (i >= 10):
            raise RunTimeError(f'Unable to find {path}')
        path = '../' + path
        i += 1


def formatTime(t):
    h = int(t / 3600)
    t -= 3600 * h;
    m = int(t / 60)
    t -= 60 * m
    s = int(t)
    return f'{h}:{m:02}:{s:02}'

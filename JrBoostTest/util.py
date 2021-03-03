import os, random, warnings
import numpy as np
import pandas as pd
import jrboost


def lorToP(a):
    return 1.0 / (1.0 + np.exp(-a))


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


def trainAndEvalExternal(trainInData, trainOutData, testInData, testOutData, optionList, lossFun, rankFun = None):

    optionCount = len(optionList)
    loss = np.zeros((optionCount,))

    maxVariableCount = max(opt.topVariableCount for opt in optionList)
    if rankFun is None:
        trainInData = trainInData[:, :maxVariableCount]
        testInData = testInData[:, :maxVariableCount]
    else:
        rankedVariables = rankFun(trainInData, trainOutData)[:maxVariableCount]
        trainInData = trainInData[:, rankedVariables]
        testInData = testInData[:, rankedVariables]

    trainer = jrboost.BoostTrainer(trainInData, trainOutData)
    loss =  trainer.trainAndEval(testInData, testOutData, optionList, lossFun)
    return loss


def trainAndEvalInternal(inData, outData, samples, foldCount, optionList, lossFun, rankFun = None):

    optionCount = len(optionList)
    loss = np.zeros((optionCount,))

    folds = stratifiedRandomFolds(outData, foldCount, samples)
    for trainSamples, testSamples in folds:

        maxVariableCount = max(opt.topVariableCount for opt in optionList)
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
        loss +=  trainer.trainAndEval(testInData, testOutData, optionList, lossFun)

    return loss


def trainAndPredictInternal(inData, outData, trainSamples, testSamples, opt, rankFun = None):

    try:
        maxVariableCount = opt.topVariableCount
    except:
        maxVariableCount = max(opt1.topVariableCount for opt1 in opt)

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
        predOutData = np.mean(np.array(predOutDataList), axis = 0)

    return predOutData

# takes average of several opts
# no support for rankFun - could be fixed
def trainAndPredictExternal(trainInData, trainOutData, testInData, opts):

    testSampleCount = testInData.shape[0]
    testOutData = np.zeros((testSampleCount,))

    trainer = jrboost.BoostTrainer(trainInData, trainOutData)
    for opt in opts:
        predictor = trainer.train(opt)
        testOutData += predictor.predict(testInData)                    # FIX THIS
    testOutData /= len(opts)
    return testOutData


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

import itertools, random, time
import numpy as np
import pandas as pd
import util
import jrboost

PROFILE = jrboost.PROFILE

#-----------------------------------------------------------------------------------------------------------------------

def main():

    threadCount = 4
    jrboost.setNumThreads(threadCount)
    jrboost.setProfile(True)
    print(f'{threadCount} threads\n')

    foldCount = 5
    usedOptCount = 10

    trainInDataFrame, trainOutDataFrame = loadTrainData(0.01)
    trainSamples = trainInDataFrame.index
    variables = trainInDataFrame.columns
    labels = trainOutDataFrame.columns
    testInDataFrame = loadTestData(0.01)
    testSamples = testInDataFrame.index
    assert (testInDataFrame.columns == variables).all()
    print(f'{len(trainSamples)} train samples, {len(testSamples)} test samples')
    print()

    trainInData = trainInDataFrame.to_numpy(dtype = np.float32)
    testInData = testInDataFrame.to_numpy(dtype = np.float32)
    for i in itertools.count():

        print(f'-------------------- {i} --------------------\n')
        t = -time.time()
        PROFILE.PUSH(PROFILE.MAIN)
        predOutDataFrame = pd.DataFrame(index = testSamples, columns = labels, dtype = np.uint64)

        for label in labels:
            print(label)
            trainOutData = trainOutDataFrame[label].to_numpy(dtype = np.uint64);
            opts = optimizeHyperParams(trainInData, trainOutData, foldCount)[:usedOptCount]
            predOutData = util.trainAndPredictExternal(trainInData, trainOutData, testInData, opts)
            predOutData = 1 / (1 + np.exp(-predOutData))
            predOutDataFrame[label] = predOutData

        PROFILE.POP()
        t += time.time()
        PROFILE.PRINT()
        print(util.formatTime(t))
        print()
        print()

        predOutDataFrame.to_csv('result.csv', sep = ',')

#-----------------------------------------------------------------------------------------------------------------------

def loadTrainData(frac = None):

    trainDataPath = util.findPath('Data/Otto/train.csv')
    trainDataFrame = pd.read_csv(trainDataPath, sep = ',', index_col = 0)
    if frac is not None:
        trainDataFrame = trainDataFrame.sample(frac = frac)

    trainOutDataSeries = trainDataFrame['target']
    trainOutDataFrame = util.oneHotEncode(trainOutDataSeries)
    trainInDataFrame = trainDataFrame.drop(['target'], axis = 1)

    return trainInDataFrame, trainOutDataFrame


def loadTestData(frac = None):

    testDataPath = util.findPath('Data/Otto/test.csv')
    testDataFrame = pd.read_csv(testDataPath, sep = ',', index_col = 0)
    if frac is not None:
        testDataFrame = testDataFrame.sample(frac = frac)

    return testDataFrame

#-----------------------------------------------------------------------------------------------------------------------

def formatOptions(opt):
    eta = opt.eta
    usr = opt.usedSampleRatio
    uvr = opt.usedVariableRatio
    return f'eta = {eta}  usr = {usr}  uvr = {uvr}'


def optimizeHyperParams(inData, outData, foldCount):

    optionsList = []
    for usr in (0.2, 0.4, 0.6, 0.8, 1.0):
        for uvr in (0.2, 0.4, 0.6, 0.8, 1.0):
            for eta in (0.05, 0.1, 0.2, 0.5, 1.0):
                opt = jrboost.BoostOptions()
                opt.iterationCount = 1000
                opt.eta = eta
                opt.usedSampleRatio = usr
                opt.usedVariableRatio = uvr
                opt.minSampleWeight = 0.001
                optionsList.append(opt)

    optionsCount = len(optionsList)
    loss = np.zeros((optionsCount,))
    folds = util.stratifiedRandomFolds(outData, foldCount)
    for trainSamples, testSamples in folds:
        loss += util.trainAndEvalInternal(inData, outData, trainSamples, testSamples, optionsList)

    sortedOptionsList = [optionsList[i] for i in np.argsort(loss)]
    return sortedOptionsList

#-----------------------------------------------------------------------------------------------------------------------

main()

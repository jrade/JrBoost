import os
os.environ['PATH'] += ';C:/Users/Rade/Anaconda3/Library/bin'

import random, time
import numpy as np
import pandas as pd
import util
import jrboost
from jrboost import PROFILE



def loadTrainData(frac = None):
    trainDataPath = r'C:/Users/Rade/Documents/Data Analysis/Data/Otto/train.csv'
    trainDataFrame = pd.read_csv(trainDataPath, sep = ',', index_col = 0)
    if frac is not None:
        trainDataFrame = trainDataFrame.sample(frac = frac)

    trainOutDataSeries = trainDataFrame['target']
    trainOutDataFrame = util.oneHotEncode(trainOutDataSeries)
    trainInDataFrame = trainDataFrame.drop(['target'], axis = 1)

    return trainInDataFrame, trainOutDataFrame


def loadTestData():
    testDataPath = r'C:/Users/Rade/Documents/Data Analysis/Data/Otto/test.csv'
    testInDataFrame = pd.read_csv(testDataPath, sep = ',', index_col = 0)
    return testInDataFrame


def formatOptions(opt):
    eta = opt.eta
    usr = opt.base.usedSampleRatio
    uvr = opt.base.usedVariableRatio
    return f'eta = {eta}  usr = {usr}  uvr = {uvr}'


def optimizeHyperParams(inData, outData, foldCount):

    optionsList = []
    for usr in (0.2, 0.4, 0.6, 0.8, 1.0):
        for uvr in (0.2, 0.4, 0.6, 0.8, 1.0):
            for eta in (0.05, 0.1, 0.2, 0.5, 1.0):
                opt = jrboost.BoostOptions()
                opt.eta = eta
                opt.iterationCount = 1000
                opt.base.usedSampleRatio = usr
                opt.base.usedVariableRatio = uvr
                optionsList.append(opt)

    optionsCount = len(optionsList)
    loss = np.zeros((optionsCount,))

    for trainSamples, testSamples in util.stratifiedRandomFolds(outData, foldCount):

        trainInData = inData[trainSamples, :]
        trainInData = np.asfortranarray(trainInData)
        trainOutData = outData[trainSamples]

        testInData = inData[testSamples, :]
        testInData = np.asfortranarray(testInData)
        testOutData = outData[testSamples]

        trainer = jrboost.BoostTrainer(trainInData, trainOutData)
        loss += trainer.trainAndEval(testInData, testOutData, optionsList)

    sortedOptionsList = [optionsList[i] for i in np.argsort(loss)]
    return sortedOptionsList


def trainAndPredict(trainInData, trainOutData, testInData, opts):

    trainInData = np.asfortranarray(trainInData)
    testInData = np.asfortranarray(testInData)
    testSampleCount = testInData.shape[0]
    testOutData = np.zeros((testSampleCount,))

    trainer = jrboost.BoostTrainer(trainInData, trainOutData)
    for opt in opts:
        predictor = trainer.train(opt)
        testOutData += predictor.predict(testInData)
    testOutData /= len(opts)
    return testOutData

#---------------------------------------------------------

trainInDataFrame, trainOutDataFrame = loadTrainData(0.01)
trainInData = trainInDataFrame.to_numpy(dtype = np.float32)
labels = trainOutDataFrame.columns

testInDataFrame = loadTestData()
testInData = testInDataFrame.to_numpy(dtype = np.float32)
testOutDataFrame = pd.DataFrame(index = testInDataFrame.index, columns = labels, dtype = np.uint64)

while True:

    t = -time.time()
    PROFILE.PUSH(PROFILE.MAIN)

    for label in labels:
        print(label)
        trainOutData = trainOutDataFrame[label].to_numpy(dtype = np.uint64);
        trainOutData = np.ascontiguousarray(trainOutData)
        opts = optimizeHyperParams(trainInData, trainOutData, 5)[:10]
        #for opt in opts:
        #    print(f'  {formatOptions(opt)}')

        testOutData = trainAndPredict(trainInData, trainOutData, testInData, opts)
        testOutData = 1 / (1 + np.exp(-testOutData))
        testOutDataFrame[label] = testOutData
        print()

    PROFILE.POP()
    t += time.time()
    PROFILE.PRINT()
    print(util.formatTime(t))

    print()
    print()

    testOutDataFrame.to_csv('result.csv', sep = ',')

print('Done!')


# 0.922  (frac = 0.01, logloss)
# 0.665  (frac = 0.1, logloss)
# 0.930 (frac = 0.01, linLoss, 10 opt)




# Class_1: eta = 0.05  usr = 0.6  uvr = 0.2
# Class_2: eta = 0.05  usr = 0.6  uvr = 0.2
# Class_3: eta = 0.1  usr = 1.0  uvr = 0.6
# Class_4: eta = 0.05  usr = 1.0  uvr = 0.2
# Class_5: eta = 0.02  usr = 0.6  uvr = 0.2
# Class_6: eta = 0.05  usr = 0.6  uvr = 0.2
# Class_7: eta = 0.05  usr = 0.8  uvr = 0.6
# Class_8: eta = 0.1  usr = 1.0  uvr = 0.2
# Class_9: eta = 0.05  usr = 0.6  uvr = 0.2
# 0:27:35


# Class_1: eta = 0.05  usr = 0.8  uvr = 0.6
# Class_2: eta = 0.05  usr = 0.6  uvr = 0.2
# Class_3: eta = 0.05  usr = 0.4  uvr = 0.2
# Class_4: eta = 0.05  usr = 0.6  uvr = 0.2
# Class_5: eta = 0.05  usr = 1.0  uvr = 0.2
# Class_6: eta = 0.05  usr = 0.8  uvr = 0.2
# Class_7: eta = 0.05  usr = 0.8  uvr = 0.2
# Class_8: eta = 0.1  usr = 1.0  uvr = 0.4
# Class_9: eta = 0.05  usr = 0.8  uvr = 0.8
# 0:27:48

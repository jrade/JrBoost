#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import itertools, random, time
import numpy as np
import pandas as pd
import util, optimize_grid, optimize_dynamic
import jrboost

PROFILE = jrboost.PROFILE

#-----------------------------------------------------------------------------------------------------------------------

cvParam = {
    'threadCount': 4,
    'profile': False,
    'dataFraction': 0.01,

    'optimizeFun': optimize_dynamic.optimize,
    'lossFun': jrboost.logLoss_lor,
    'innerFoldCount': 5,

    'boostParamValues': {
        'iterationCount': [1000],
        'eta':  [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
        'usedSampleRatio': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        'usedVariableRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'minNodeSize': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        'minSampleWeight': [0.001],
    },

    'bestOptionCount': 10,
    'bagSize': 1,

    'populationCount': 100,
    'survivorCount': 50,
    'cycleCount': 10,
}

#-----------------------------------------------------------------------------------------------------------------------

def main():

    print(cvParam)
    print()

    threadCount = cvParam['threadCount']
    profile = cvParam['profile']
    dataFraction = cvParam['dataFraction']
    optimizeFun = cvParam['optimizeFun']
    lossFun = cvParam['lossFun']
    innerFoldCount = cvParam['innerFoldCount']

    jrboost.setProfile(profile)
    jrboost.setNumThreads(threadCount)
    print(f'{threadCount} threads\n')

    trainInDataFrame, trainOutDataFrame = loadTrainData(dataFraction)
    trainSamples = trainInDataFrame.index
    variables = trainInDataFrame.columns
    labels = trainOutDataFrame.columns

    testInDataFrame = loadTestData(0.01)
    testSamples = testInDataFrame.index
    assert (testInDataFrame.columns == variables).all

    print(f'{len(trainSamples)} train samples, {len(testSamples)} test samples\n')

    trainInData = trainInDataFrame.to_numpy(dtype = np.float32)
    testInData = testInDataFrame.to_numpy(dtype = np.float32)
    for i in itertools.count():

        print(f'-------------------- {i} --------------------\n')
        t = -time.time()
        PROFILE.PUSH(PROFILE.MAIN)
        predOutDataFrame = pd.DataFrame(index = testSamples, columns = labels, dtype = np.uint64)

        for label in labels:
            trainOutData = trainOutDataFrame[label].to_numpy(dtype = np.uint64);

            bestOptList = optimizeFun(
                cvParam,
                lambda optionList: 
                    trainAndEval(trainInData, trainOutData, innerFoldCount, optionList, lossFun)
            )                 

            print(f'{label}: {formatOptions(bestOptList[0])}')
            predOutData = trainAndPredict(trainInData, trainOutData, testInData, bestOptList)
            predOutDataFrame[label] = predOutData

        PROFILE.POP()
        t += time.time()
        PROFILE.PRINT()
        print(util.formatTime(t))
        print()
        print()

        predOuDataFrame = util.lorToProb(predOutDataFrame)
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


def trainAndEval(inData, outData, foldCount, optionList, lossFun):

    optionCount = len(optionList)
    loss = np.zeros((optionCount,))
    folds = util.stratifiedRandomFolds(outData, foldCount)
    for trainSamples, testSamples in folds:

        trainInData = inData[trainSamples, :]
        testInData = inData[testSamples, :]
        trainOutData = outData[trainSamples]
        testOutData = outData[testSamples]

        trainer = jrboost.BoostTrainer(trainInData, trainOutData)
        loss += trainer.trainAndEval(testInData, testOutData, optionList, lossFun)

    return loss


def trainAndPredict(trainInData, trainOutData, testInData, bestOptList):

    predOutDataList = []
    trainer = jrboost.BoostTrainer(trainInData, trainOutData)
    for opt in bestOptList:
        predictor = trainer.train(opt)
        predOutDataList.append(predictor.predict(testInData))
    predOutData = np.median(np.array(predOutDataList), axis = 0)
    return predOutData


def formatOptions(opt):
    eta = opt.eta
    usr = opt.usedSampleRatio
    uvr = opt.usedVariableRatio
    mns = opt.minNodeSize
    return f'eta = {eta:.2f}  usr = {usr:.1f}  uvr = {uvr:.1f}  mns = {mns:2}'

#-----------------------------------------------------------------------------------------------------------------------

main()

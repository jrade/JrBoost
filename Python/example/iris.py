#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import itertools, random, sys, time
import numpy as np
import pandas as pd
import util, optimize_grid, optimize_dynamic
import jrboost

PROFILE = jrboost.PROFILE

#-----------------------------------------------------------------------------------------------------------------------

cvParam = {
    'threadCount': 4,
    'profile': True,

    'optimizeFun': optimize_grid.optimize,
    'lossFun': jrboost.negAuc,

    'boostParamValues': {
        'method': [jrboost.BoostOptions.Logit],
        'gamma': [0.5],
        'iterationCount': [1000],
        'eta':  [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
        'usedSampleRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'usedVariableRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'minNodeSize': [1, 2, 3, 4, 6, 8, 10],
        'minSampleWeight': [0.001],
    },

    'outerFoldCount': 5,
    'innerFoldCount': 5,
    'bestOptionCount': 1,
    'bagSize': None,
    'ultraBoost': None,

    'populationCount': 100,
    'survivorCount': 50,
    'cycleCount': 3,
}

#-----------------------------------------------------------------------------------------------------------------------

def main():

    print(cvParam)
    print()

    threadCount = cvParam['threadCount']
    profile = cvParam['profile']
    outerFoldCount = cvParam['outerFoldCount']
    innerFoldCount = cvParam['innerFoldCount']
    optimizeFun = cvParam['optimizeFun']
    lossFun = cvParam['lossFun']

    jrboost.setProfile(profile)
    jrboost.setNumThreads(threadCount)
    print(f'{threadCount} threads\n')

    inDataFrame, outDataSeries = loadData()
    outDataFrame = util.oneHotEncode(outDataSeries)
    samples = outDataFrame.index
    variables = inDataFrame.columns
    labels = outDataFrame.columns
    print(f'{len(samples)} samples, {len(variables)} variables')

    inData = inDataFrame.to_numpy(dtype = np.float32)
    for i in itertools.count():

        print(f'-------------------- {i} --------------------\n')
        t = -time.time()
        PROFILE.PUSH(PROFILE.MAIN)
        predOutDataFrame = pd.DataFrame(index = samples, columns = labels, dtype = np.float64)

        for label in labels:
            print(label)
            outData = outDataFrame[label].to_numpy(dtype = np.uint64)

            predOutData = np.empty((len(samples),))
            folds = util.stratifiedRandomFolds(outData, outerFoldCount)
            for trainSamples, testSamples in folds:
                bestOptList = optimizeFun(
                    cvParam,
                    lambda optionList: 
                        trainAndEval(inData, outData, trainSamples, innerFoldCount, optionList, lossFun)
                )                 
                print(formatOptions(bestOptList[0]))
                predOutData[testSamples] = trainAndPredict(inData, outData, trainSamples, testSamples, bestOptList)       
           
            predOutDataFrame[label] = predOutData
            print()

        print()
        predOutDataSeries = predOutDataFrame.idxmax(axis = 1)
        confusionFrame = pd.DataFrame(index = labels, columns = labels, data = 0)
        for sample in samples:
            confusionFrame.loc[outDataSeries[sample], predOutDataSeries[sample]] += 1

        PROFILE.POP()
        t += time.time()
        PROFILE.PRINT()
        print()
        print(confusionFrame)
        print()
        print(util.formatTime(t))
        print()

#-----------------------------------------------------------------------------------------------------------------------

def loadData():
    dataPath = util.findPath('/Data/Iris/Iris.csv')
    dataFrame = pd.read_csv(dataPath, sep = ',', index_col = 0)
    outDataSeries = dataFrame['Species']
    inDataFrame = dataFrame.drop(['Species'], axis = 1)
    return inDataFrame, outDataSeries


def trainAndEval(inData, outData, samples, foldCount, optionList, lossFun):

    optionCount = len(optionList)
    loss = np.zeros((optionCount,))
    folds = util.stratifiedRandomFolds(outData, foldCount, samples)
    for trainSamples, testSamples in folds:

        trainInData = inData[trainSamples, :]
        testInData = inData[testSamples, :]
        trainOutData = outData[trainSamples]
        testOutData = outData[testSamples]

        trainer = jrboost.BoostTrainer(trainInData, trainOutData)
        loss +=  trainer.trainAndEval(testInData, testOutData, optionList, lossFun)

    return loss


def trainAndPredict(inData, outData, trainSamples, testSamples, opt, rankFun = None):

    trainInData = inData[trainSamples, :]
    testInData = inData[testSamples, :]
    trainOutData = outData[trainSamples]

    trainer = jrboost.BoostTrainer(trainInData, trainOutData)

    predOutDataList = []
    for opt1 in opt:
        predictor = trainer.train(opt1);
        predOutData = predictor.predict(testInData);
        predOutDataList.append(predOutData)
    predOutData = np.median(np.array(predOutDataList), axis = 0)
    return predOutData


def formatOptions(opt):
    ic = opt.iterationCount
    eta  = opt.eta
    usr = opt.usedSampleRatio
    uvr = opt.usedVariableRatio
    mns = opt.minNodeSize
    return f'ic = {ic}  eta = {eta:.2f}  usr = {usr:.1f}  uvr = {uvr:.1f}  mns = {mns:2}'

#-----------------------------------------------------------------------------------------------------------------------

main()

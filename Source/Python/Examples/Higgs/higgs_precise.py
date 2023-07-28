#  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import datetime, itertools, math, os, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jrboost

#-----------------------------------------------------------------------------------------------------------------------

param = {
    'threadCount': os.cpu_count() // 2,
    'sampleRatio':  0.7,
    'subsample': 0.01,
    'repCount': 10,
    'smoothThreshold': False,
}

trainParam = {
    'repCount': 1,
    'foldCount': 5,
    'targetLossFun': jrboost.LogLoss(0.001),
    'minimizeAlgorithm': jrboost.minimizePopulation,

    'minimizeParam': {
        'populationCount': 100,
        'survivorCount': 50,
        'cycleCount': 2,
        'bestCount': 10,
    },

    'boostParamGrid': {
        #'minRelSampleWeight': [0.001],
        #'iterationCount': [1000],
        'iterationCount': [300],
        'eta':  [0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
        'usedSampleRatio': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'usedVariableRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'maxTreeDepth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'minNodeSize': [100, 150, 200, 300, 500, 700, 1000],
        #'selectVariablesByLevel': [True],
    },
}

outDirPath = None
log = None
iterationIndex = None

#-----------------------------------------------------------------------------------------------------------------------

def main():

    global outDirPath
    timeStamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    subsample = param.get('subsample', None)
    repCount = param.get('repCount', None)
    outDirPath = f'../../Higgs output/Train C {timeStamp} {subsample} {repCount}/'

    assert not os.path.exists(outDirPath)
    os.makedirs(outDirPath)

    threadCount = param['threadCount']
    jrboost.setThreadCount(threadCount)

    trainInDataFrame, trainOutDataSeries, trainWeightSeries, testInDataFrame = loadData()
    trainInData = trainInDataFrame.to_numpy(dtype = np.float32)
    trainOutData = trainOutDataSeries.to_numpy(dtype = np.uint8)
    trainWeights = trainWeightSeries.to_numpy(dtype = np.float64)
    testInData = testInDataFrame.to_numpy(dtype = np.float32)

    logFilePath = outDirPath + f'Log.txt'
    with open(logFilePath, 'w', 1) as logFile:
        def _log(msg = '', end = '\n'): print(msg, end = end); logFile.write(msg + end)

        global log
        log = _log

        log(jrboost.formatParam(param) + '\n')
        log(jrboost.formatParam(trainParam) + '\n')

        global iterationIndex
        for iterationIndex in itertools.count():
            log(f'----------------------- {iterationIndex} ---------------------------\n')


            t = -time.time()
            jrboost.PROFILE.START()

            predictor, threshold = trainC(trainInData, trainOutData, trainWeights)

            t += time.time()
            log(jrboost.PROFILE.STOP())
            log(formatTime(t) + '\n')

            # save result

            predictorFilePath = outDirPath + f'Predictor {iterationIndex}.jrboost'
            predictor.save(predictorFilePath)

            t = -time.time()
            testPredData = predictor.predict(testInData);
            t += time.time()
            log(formatTime(t) + '\n')

            testPredDataSeries = pd.Series(
                index = testInDataFrame.index,
                data = testPredData
            )
            submissionDataFrame = pd.DataFrame({
                'RankOrder': testPredDataSeries.rank(method = 'first', ascending = False).astype(int),
                'Class': (testPredDataSeries >= threshold).map({False: 'b', True: 's'}) 
            })
            submissionFilePath = outDirPath + f'Submission {iterationIndex}.csv'
            submissionDataFrame.to_csv(submissionFilePath, sep = ',')

#-----------------------------------------------------------------------------------------------

def loadData():

    dataDirPath = '../../../Data/Higgs/'

    trainDataFilePath = dataDirPath + 'training.csv'
    trainDataFrame = pd.read_csv(trainDataFilePath, sep = ',', index_col = 0)
    trainDataFrame.index.name = 'EventId'

    trainOutDataSeries = trainDataFrame['Label'].map({'b': 0, 's': 1})
    trainWeightSeries = trainDataFrame['Weight']
    trainInDataFrame = trainDataFrame.drop(['Label', 'Weight'], axis = 1)

    testDataFilePath = dataDirPath + 'test.csv'
    testInDataFrame = pd.read_csv(testDataFilePath, sep = ',', index_col = 0)
    testInDataFrame.index.name = 'EventId'

    return trainInDataFrame, trainOutDataSeries, trainWeightSeries, testInDataFrame

#-----------------------------------------------------------------------------------------------------------------------

def trainA(inData, outData, weights):

    sampleRatio = param['sampleRatio']
    inData1, outData1, weights1, inData2, outData2, weights2 = _splitData(inData, outData, weights, sampleRatio)
    weights1 *= weights.sum() / weights1.sum()
    weights2 *= weights.sum() / weights2.sum()

    bestBoostParams = jrboost.optimizeHyperParam(inData1, outData1, trainParam, weights = weights1)
    trainer = jrboost.BoostTrainer(inData1, outData1, weights = weights1)
    predictor = jrboost.Predictor.createEnsemble(jrboost.parallelTrain(trainer, bestBoostParams))
    log(formatBoostParams(bestBoostParams))

    predData2 = predictor.predict(inData2)
    threshold = optimalAmsThreshold(outData2, predData, weights2)
    log(f'threshold = {threshold}')

    return predictor, threshold


def trainB(inData, outData, weights):

    bestBoostParams = jrboost.optimizeHyperParam(inData, outData, trainParam, weights = weights)
    log(formatBoostParams(bestBoostParams))

    predictors = []
    thresholds = []

    for i, bp in enumerate(bestBoostParams):

        sampleRatio = param['sampleRatio']
        inData1, outData1, weights1, inData2, outData2, weights2 = _splitData(inData, outData, weights, sampleRatio)
        weights1 *= weights.sum() / weights1.sum()
        weights2 *= weights.sum() / weights2.sum()

        trainer = jrboost.BoostTrainer(inData1, outData1, weights = weights1)
        predictor = trainer.train(bp)
        predData2 = predictor.predict(inData2)
        threshold = optimalAmsThreshold(outData2, predData2, weights2, i)

        predictors.append(predictor)
        thresholds.append(threshold)

    predictor = jrboost.Predictor.createEnsemble(predictors)
    threshold = np.mean(thresholds)

    log(f'threshold = {threshold}')

    return predictor, threshold


def trainC(inData, outData, weights):

    subsample = param.get('subsample', 1.0)
    repCount = param.get('repCount', 1)
    sampleRatio = param['sampleRatio']

    subInData, subOutData, subWeights = _subData(inData, outData, weights, subsample)
    bestBoostParams = jrboost.optimizeHyperParam(subInData, subOutData, trainParam, weights = subWeights)
    log(formatBoostParams(bestBoostParams))

    outerPredictors = []
    thresholds = []

    for i in range(repCount):

        inData1, outData1, weights1, inData2, outData2, weights2 = _splitData(inData, outData, weights, sampleRatio)
        weights1 *= weights.sum() / weights1.sum()
        weights2 *= weights.sum() / weights2.sum()

        subInData1, subOutData1, subWeights1 = _subData(inData1, outData1, weights1, subsample)
        trainer = jrboost.BoostTrainer(subInData1, subOutData1, weights = subWeights1)
        innerPredictors = jrboost.parallelTrain(trainer, bestBoostParams)
        innerPredictor = jrboost.Predictor.createEnsemble(innerPredictors)
        outerPredictors.append(innerPredictor)

        predData2 = innerPredictor.predict(inData2)
        threshold = optimalAmsThreshold(outData2, predData2, weights2, i)
        thresholds.append(threshold)

    tmp = list(zip(outerPredictors, thresholds))
    tmp.sort(key = lambda x: x[1])
    trim = (repCount + 5) // 10
    tmp = tmp[trim: -trim]
    outerPredictors, thresholds = zip(*tmp)

    outerPredictor = jrboost.Predictor.createEnsemble(outerPredictors)
    threshold = np.mean(thresholds)
    log(f'threshold = {threshold}\n')

    return outerPredictor, threshold

#-----------------------------------------------------------------------------------------------------------------------

def _subData(inData, outData,  weights, ratio):
    if ratio == 1.0:
       return inData, outData, weights
    samples, _ = jrboost.stratifiedRandomSplit(outData, ratio)
    return inData[samples, :], outData[samples], weights[samples]

def _splitData(inData, outData,  weights, ratio):
    samples1, samples2 = jrboost.stratifiedRandomSplit(outData, ratio)
    return inData[samples1, :], outData[samples1], weights[samples1], inData[samples2, :], outData[samples2], weights[samples2]

#-----------------------------------------------------------------------------------------------------------------------

def optimalAmsThreshold(outData, predData, weights, plotIndex = None):

    i = np.flip(np.argsort(predData))
    outData = outData[i]
    predData = predData[i]
    weights = weights[i]

    truePos = 0.0
    falsePos = 0.0
    smoothTruePos = 0.0
    smoothFalsePos = 0.0

    thresholds = []
    scores = []
    smoothScores = []

    sampleCount = len(outData)

    for i in range(0, sampleCount - 1):

        outValue = outData[i]
        predValue = predData[i]
        weight = weights[i]

        truePos += outValue * weight
        falsePos += (1.0 - outValue) * weight
        smoothTruePos += predValue * weight
        smoothFalsePos += (1.0 - predValue) * weight

        nextPredValue = predData[i + 1]
        if predValue == nextPredValue: continue

        thresholds.append((predValue + nextPredValue) / 2.0)
        scores.append(_amsScore(truePos, falsePos))
        smoothScores.append(_amsScore(smoothTruePos, smoothFalsePos))

    j = np.argmax(scores)
    threshold = thresholds[j]
    j = np.argmax(smoothScores)
    smoothThreshold = thresholds[j]

    plt.clf()
    plt.scatter(x = thresholds, y = scores, s = 1, c = 'r')
    plt.scatter(x = thresholds, y = smoothScores, s = 1, c = 'k')
    plt.xlim(0.0, 0.1)
    plt.ylim(0.0, 5.0)
    plt.vlines(threshold, 0.0, 5.0, 'brown')
    plt.vlines(smoothThreshold, 0.0, 5.0, 'gray')
    plt.show(block = False)
    plt.pause(1.0)
    if plotIndex is None:
        plotFilePath = outDirPath + f'Plot {iterationIndex}.png'
    else:
        plotFilePath = outDirPath + f'Plot {iterationIndex}-{plotIndex}.png'
    plt.savefig(plotFilePath)

    return smoothThreshold if param.get('smoothThreshold', False) else threshold


def _amsScore(s, b):
    b_r = 10.0
    return math.sqrt( 2.0 * (
        (s + b + b_r) 
        * math.log (1.0 + s / (b + b_r)) 
        - s
    ))

#-----------------------------------------------------------------------------------------------------------------------

def formatBoostParams(boostParams):
    return formatBoostParam(jrboost.medianBoostParam(boostParams))

def formatBoostParam(opt):
    eta = opt['eta']
    usr = opt['usedSampleRatio']
    uvr = opt['usedVariableRatio']
    mns = opt['minNodeSize']
    md = opt['maxTreeDepth']
    return f'eta = {eta}  usr = {usr}  uvr = {uvr}  mns = {mns}  md = {md}'

def formatScore(score, precision = 4):
    return '(' + ', '.join((f'{x:.{precision}f}' for x in score)) + ')'

def formatTime(t):
    h = int(t / 3600)
    t -= 3600 * h;
    m = int(t / 60)
    t -= 60 * m
    s = int(t)
    return f'{h}:{m:02}:{s:02}'

#-----------------------------------------------------------------------------------------------------------------------

main()

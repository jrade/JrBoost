#  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import math, os, time
import numpy as np
import pandas as pd
import jrboost

#-----------------------------------------------------------------------------------------------------------------------

dataDirPath = '../../../Data/Higgs/'

param = {
    'threadCount': os.cpu_count() // 2,
    'sampleRatio':  2/3,
}

boostParam = {
    'iterationCount': 1000,
    'eta': 0.03,
    'usedSampleRatio': 0.8,
    'usedVariableRatio': 0.3,
    'maxTreeDepth': 8,
    'minNodeSize': 300,
}

print(param)
print()

print(boostParam)
print()

#-----------------------------------------------------------------------------------------------------------------------

def main():

    threadCount = param['threadCount']
    trainFraction = param.get('trainFraction', None)

    jrboost.setThreadCount(threadCount)

    trainInDataFrame, trainOutDataSeries, trainWeightSeries = loadTrainData()

    trainInData = trainInDataFrame.to_numpy(dtype = np.float32)
    trainOutData = trainOutDataSeries.to_numpy(dtype = np.uint8)
    trainWeights = trainWeightSeries.to_numpy(dtype = np.float64)

    t = -time.time()
    jrboost.PROFILE.START()

    predictor, threshold = train(trainInData, trainOutData, trainWeights)

    t += time.time()
    print(jrboost.PROFILE.STOP())
    print(formatTime(t) + '\n')

    print(f'threshold = {threshold}')

    saveSubmission(predictor, threshold)

#-----------------------------------------------------------------------------------------------------------------------

def train(inData, outData, weights):

    sampleRatio = param['sampleRatio']
    inData1, outData1, weights1, inData2, outData2, weights2 = _splitData(inData, outData, weights, sampleRatio)
    weights1 *= weights.sum() / weights1.sum()
    weights2 *= weights.sum() / weights2.sum()

    trainer = jrboost.BoostTrainer(inData1, outData1, weights = weights1)
    predictor = trainer.train(boostParam)

    predOutData2 = predictor.predict(inData2)
    threshold, _ = optimalTheshold(outData2, predOutData2, weights2)

    return predictor, threshold

#-----------------------------------------------------------------------------------------------

def loadTrainData():

    trainDataFilePath = dataDirPath + 'training.csv'
    trainDataFrame = pd.read_csv(trainDataFilePath, sep = ',', index_col = 0)
    trainDataFrame.index.name = 'EventId'

    trainOutDataSeries = trainDataFrame['Label'].map({'b': 0, 's': 1})
    trainWeightSeries = trainDataFrame['Weight']
    trainInDataFrame = trainDataFrame.drop(['Label', 'Weight'], axis = 1)

    return trainInDataFrame, trainOutDataSeries, trainWeightSeries


def saveSubmission(predictor, threshold):

    testDataFilePath = dataDirPath + 'test.csv'
    testInDataFrame = pd.read_csv(testDataFilePath, sep = ',', index_col = 0)
    testInDataFrame.index.name = 'EventId'

    testInData = testInDataFrame.to_numpy(dtype = np.float32)

    testPredData = predictor.predict(testInData)

    testPredDataSeries = pd.Series(
        index = testInDataFrame.index,
        data = testPredData
    )

    submissionDataFrame = pd.DataFrame({
        'RankOrder': testPredDataSeries.rank(method = 'first', ascending = False).astype(int),
        'Class': (testPredDataSeries >= threshold).map({False: 'b', True: 's'}) 
    })

    submissionDataFrame.to_csv(dataDirPath + 'submission.csv', sep = ',')

 
def _splitData(inData, outData,  weights, ratio):
    samples1, samples2 = jrboost.stratifiedRandomSplit(outData, ratio)
    return inData[samples1, :], outData[samples1], weights[samples1], inData[samples2, :], outData[samples2], weights[samples2]


def formatTime(t):
    h = int(t / 3600)
    t -= 3600 * h;
    m = int(t / 60)
    t -= 60 * m
    s = int(t)
    return f'{h}:{m:02}:{s:02}'

#-----------------------------------------------------------------------------------------------------------------------

def amsScore_(s, b):

    b_r = 10.0
    return math.sqrt( 2 * (
		(s + b + b_r) * math.log (1 +  (s / (b + b_r)))
		-s
	))


def amsScore(outData, predData, weights):

    truePos = np.sum(outData * predData * weights)
    trueNeg = np.sum((1 - outData) * predData * weights)
    return amsScore_(truePos, trueNeg)


def optimalTheshold(outData, predData, weights):

    a = sorted(list(zip(outData, predData, weights)), key = lambda x: -x[1])

    truePos = 0.0
    falsePos = 0.0

    bestScore = amsScore_(truePos, falsePos)
    bestI = -1

    for i, (outValue, _, weight) in enumerate(a):

        if outValue:
            truePos += weight
        else:
            falsePos += weight

        score = amsScore_(truePos, falsePos)
        if score <= bestScore: continue

        bestScore = score
        bestI = i


    assert bestI != -1
    assert bestI != len(a) - 1
    assert a[bestI][1] != a[bestI + 1][1]     # How to handle this?     

    bestTheshold = (a[bestI][1] + a[bestI + 1][1]) / 2.0 

    return bestTheshold, bestScore

#-------------------------------------------

main()

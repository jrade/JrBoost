
#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import math, os, random, time
import numpy as np
import pandas as pd
import jrboost

PROFILE = jrboost.PROFILE

#-----------------------------------------------------------------------------------------------------------------------

param = {
    'threadCount': os.cpu_count() - 2,
    'profile': True,
    'trainFraction': 0.5,
}

trainParam = {

    'minimizeAlgorithm': jrboost.minimizePopulation,
    'lossFun': jrboost.logLossWeighted,

    'boostParamGrid': {
        'iterationCount': [1000],
        'eta':  [0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
        'usedSampleRatio': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        'usedVariableRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'minNodeSize': [10, 20, 50, 100, 200, 500, 1000],                  # increase with train fraction 1.0?
        #'minRelSampleWeight': [0.000001],
        'minRelSampleWeight': [0.001],
        'maxDepth': [1, 2, 3, 4]
    },

    'minimizeParam': {
        'populationCount': 100,
        'survivorCount': 50,
        'cycleCount': 2,
        'bestCount': 10,
    }
}

print(param)
print()
print(trainParam)
print()

#-----------------------------------------------------------------------------------------------------------------------

def main():

    threadCount = param['threadCount']
    profile = param['profile']
    trainFraction = param.get('trainFraction', None)

    jrboost.setThreadCount(threadCount)

    (trainInDataFrame, trainOutDataSeries, trainWeightSeries,
        testInDataFrame, testOutDataSeries, testWeightSeries,
        validationInDataFrame, validationOutDataSeries, validationWeightSeries
    ) = loadData(trainFrac = trainFraction)

    trainInData = trainInDataFrame.to_numpy(dtype = np.float32)
    trainOutData = trainOutDataSeries.to_numpy(dtype = np.uint64)
    trainWeights = trainWeightSeries.to_numpy(dtype = np.float64)
    testInData = testInDataFrame.to_numpy(dtype = np.float32)
    testOutData = testOutDataSeries.to_numpy(dtype = np.uint64)
    testWeights = testWeightSeries.to_numpy(dtype = np.float64)
    validationInData = validationInDataFrame.to_numpy(dtype = np.float32)
    validationOutData = validationOutDataSeries.to_numpy(dtype = np.uint64)
    validationWeights = validationWeightSeries.to_numpy(dtype = np.float64)

    adjTrainWeights = np.empty_like(trainWeights)
    adjTrainWeights[trainOutData == 0] = trainWeights[trainOutData == 0] / trainWeights[trainOutData == 0].mean()
    adjTrainWeights[trainOutData == 1] = trainWeights[trainOutData == 1] / trainWeights[trainOutData == 1].mean()
    
    adjTestWeights = np.empty_like(testWeights)
    adjTestWeights[testOutData == 0] = testWeights[testOutData == 0] / testWeights[testOutData == 0].mean()
    adjTestWeights[testOutData == 1] = testWeights[testOutData == 1] / testWeights[testOutData == 1].mean()
    
    adjValidationWeights = np.empty_like(validationWeights)
    adjValidationWeights[validationOutData == 0] = validationWeights[validationOutData == 0] / validationWeights[validationOutData == 0].mean()
    adjValidationWeights[validationOutData == 1] = validationWeights[validationOutData == 1] / validationWeights[validationOutData == 1].mean()
 
    print(pd.DataFrame(
        index = ['Train', 'Test', 'Validation'],
        columns = ['Total', 'Neg.', 'Pos.', 'Pos. Ratio', 'Weight'],
        data = [
            [len(trainOutData), (1 - trainOutData).sum(), trainOutData.sum(), trainOutData.sum() / len(trainOutData), trainWeightSeries.sum()],
            [len(testOutData), (1 - testOutData).sum(), testOutData.sum(), testOutData.sum() / len(testOutData), testWeightSeries.sum()],
            [len(validationOutData), (1 - validationOutData).sum(), validationOutData.sum(), validationOutData.sum() / len(validationOutData), validationWeightSeries.sum()],
        ]
    ))
    print()

    # train predictor ..........................................................

    t = -time.time()
    if profile: PROFILE.START()

    predictor, cutoff, msg = train(
        trainInData, trainOutData, adjTrainWeights,
       testInData, testOutData, adjTestWeights)
    print(msg + '\n')

    t += time.time()
    if profile: print(PROFILE.STOP() + '\n')

    # score ...................................................................

    validationPredData = predictor.predict(validationInData)
    print(f'cutoff = {cutoff}\n')

    lossFun = trainParam['lossFun']
    loss = lossFun(validationOutData, validationPredData, adjValidationWeights)
    print(f'log loss: {loss}\n')

    score = amsScore(validationOutData, validationPredData >= cutoff, validationWeights)
    print(f'AMS = {score}')

    # create and save  .........................................................

    testPredData = predictor.predict(testInData)

    testPredDataSeries = pd.Series(index = testOutDataSeries.index, data = testPredData)
    validationPredDataSeries = pd.Series(index = validationOutDataSeries.index, data = validationPredData)
    submissionPredDataSeries = pd.concat((testPredDataSeries, validationPredDataSeries)).sort_index()
    submissionPredData = submissionPredDataSeries.to_numpy()
    submissionPredRank = rank(submissionPredData) + 1
    submissionPredClass = np.where(submissionPredData >= cutoff, 's', 'b')

    submissionDataFrame = pd.DataFrame(
        index = submissionPredDataSeries.index,
        data = { 'RankOrder': submissionPredRank, 'Class': submissionPredClass, }
    )
    submissionDataFrame.to_csv('../../Higgs Submission.csv', sep = ',')

#-----------------------------------------------------------------------------------------------------------------------

def train(trainInData, trainOutData, trainWeights, testInData, testOutData, testWeights):

    optimizeFun = trainParam['minimizeAlgorithm']
    lossFun = trainParam['lossFun']

    # determine the best hyperparameters
    boostParamGrid = trainParam['boostParamGrid']
    minimizeParam = trainParam['minimizeParam']
    bestOptList = optimizeFun(
        lambda optionList: evaluateBoostParam(
            trainInData, trainOutData, trainWeights, testInData, testOutData, testWeights, optionList, lossFun),
        boostParamGrid,
        minimizeParam
    )
    msg = formatOptions(bestOptList[0])

    # build predictor
    trainer = jrboost.BoostTrainer(trainInData, trainOutData, trainWeights)
    predictor = jrboost.EnsemblePredictor(jrboost.parallelTrain(trainer, bestOptList))

    predOutData = predictor.predict(testInData)

    estCutoff, _ = optimalCutoff(testOutData, predOutData, testWeights)

    #estSignalRatio = sum(trainOutData) / len(trainOutData)
    #estCutoff = np.quantile(predOutData, 1.0 - estSignalRatio)

    return predictor, estCutoff, msg


def evaluateBoostParam(
    trainInData, trainOutData, trainWeights,
    testInData, testOutData, testWeights,
    optionList, lossFun
):
    optionCount = len(optionList)
    loss = np.zeros((optionCount,))

    trainer = jrboost.BoostTrainer(trainInData, trainOutData, trainWeights)
    loss =  jrboost.parallelTrainAndEvalWeighted(trainer, optionList, testInData, testOutData, testWeights, lossFun)
    return loss

#-----------------------------------------------------------------------------------------------

def loadData(trainFrac = None):

    dataFilePath = 'C:/Users/Rade/Documents/Data Analysis/Kaggle/Higgs Challenge/Data/atlas-higgs-challenge-2014-v2.csv'
    dataFrame = pd.read_csv(dataFilePath, sep = ',', index_col = 0)

    trainSamples = dataFrame.index[dataFrame['KaggleSet'] == 't']
    if trainFrac is not None and trainFrac != 1.0:
        trainSampleCount = len(trainSamples)
        trainSamples = pd.Index(random.sample(
            trainSamples.tolist(),
            int(trainFrac * trainSampleCount + 0.5)
        ))
    testSamples = dataFrame.index[dataFrame['KaggleSet'] == 'b']
    validationSamples = dataFrame.index[dataFrame['KaggleSet'] == 'v']

    outDataSeries = pd.Series(index = dataFrame.index, data = 0)
    outDataSeries[dataFrame['Label'] == 's'] = 1
    weightSeries = dataFrame['KaggleWeight']
    inDataFrame = dataFrame.drop(['Label', 'Weight', 'KaggleSet', 'KaggleWeight'], axis = 1)

    trainInDataFrame = inDataFrame.loc[trainSamples, :]
    trainOutDataSeries = outDataSeries[trainSamples]
    trainWeightSeries = weightSeries[trainSamples]

    testInDataFrame = inDataFrame.loc[testSamples, :]
    testOutDataSeries = outDataSeries[testSamples]
    testWeightSeries = weightSeries[testSamples]

    validationInDataFrame = inDataFrame.loc[validationSamples, :]
    validationOutDataSeries = outDataSeries[validationSamples]
    validationWeightSeries = weightSeries[validationSamples]

    return (
        trainInDataFrame, trainOutDataSeries, trainWeightSeries,
        testInDataFrame, testOutDataSeries, testWeightSeries,
        validationInDataFrame, validationOutDataSeries, validationWeightSeries
    )
  
#-----------------------------------------------------------------------------------------------------------------------

def formatOptions(opt):
    eta = opt['eta']
    usr = opt['usedSampleRatio']
    uvr = opt['usedVariableRatio']
    mns = opt['minNodeSize']
    md = opt['maxDepth']
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

def rank(data):
    temp = data.argsort()
    ranks1 = np.empty_like(temp)
    ranks1[temp] = np.arange(len(temp))
    return ranks1

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


def optimalCutoff(outData, predData, weights):

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

    bestCutoff = (a[bestI][1] + a[bestI + 1][1]) / 2.0 

    return bestCutoff, bestScore


main()

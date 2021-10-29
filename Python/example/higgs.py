#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import datetime, gc, math, os, random, time
import numpy as np
import pandas as pd
import jrboost

PROFILE = jrboost.PROFILE

#-----------------------------------------------------------------------------------------------------------------------

param = {
    'threadCount': os.cpu_count() // 2,
    'profile': True,
    #'trainFraction': 0.1,
}

trainParam = {

    'minimizeAlgorithm': jrboost.minimizePopulation,
    'foldCount': 5,
    #'lossFun': jrboost.negAucWeighted,
    'lossFun': jrboost.LogLossWeighted(0.001),
    #'lossFun': lambda a, b, c: -optimalCutoff(a, b, c)[1],

    'boostParamGrid': {
        #'minRelSampleWeight': [0.001],          #??????????????????????????+
        #'iterationCount': [1000],
        'iterationCount': [300],
        'eta':  [0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
        'usedSampleRatio': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'usedVariableRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'maxDepth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'minNodeSize': [100, 150, 200, 300, 500, 700, 1000],
        #'pruneFactor': [0.0, 0.1, 0.2, 0.5],
    },

    'minimizeParam': {
        'populationCount': 100,
        'survivorCount': 50,
        'cycleCount': 2,
        'bestCount': 10,
    }
}

#-----------------------------------------------------------------------------------------------------------------------

def main():

    logFilePath = f'../Log OptTree {datetime.datetime.now().strftime("%y%m%d-%H%M%S")}.txt'
    with open(logFilePath, 'w', 1) as logFile:
        def log(msg = ''): print(msg); logFile.write(msg + '\n')

        log(f'Parameters: {param}\n')
        log(f'Training parameters:{trainParam}\n')

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
 
        log(pd.DataFrame(
            index = ['Train', 'Test', 'Validation'],
            columns = ['Total', 'Neg.', 'Pos.', 'Pos. Ratio', 'Weight'],
            data = [
                [len(trainOutData), (1 - trainOutData).sum(), trainOutData.sum(), trainOutData.sum() / len(trainOutData), trainWeightSeries.sum()],
                [len(testOutData), (1 - testOutData).sum(), testOutData.sum(), testOutData.sum() / len(testOutData), testWeightSeries.sum()],
                [len(validationOutData), (1 - validationOutData).sum(), validationOutData.sum(), validationOutData.sum() / len(validationOutData), validationWeightSeries.sum()],
            ]
        ).to_string() + '\n')

        # train predictor ..........................................................

        t = -time.time()
        if profile: PROFILE.START()

        predictor, cutoff, msg = train(trainInData, trainOutData, trainWeights)
        log(msg + '\n')

        t += time.time()
        if profile: log(PROFILE.STOP() + '\n')
        log(formatTime(t) + '\n')

        log(f'cutoff = {cutoff}\n')

        # score ...................................................................

        testPredData = predictor.predict(testInData)
        score = amsScore(testOutData, testPredData, testWeights, cutoff)
        log(f'test AMS = {score}')

        validationPredData = predictor.predict(validationInData)
        score = amsScore(validationOutData, validationPredData, validationWeights, cutoff)
        log(f'validation AMS = {score}')

        # create and save  .........................................................

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

def train(inData, outData, weights):

    optimizeFun = trainParam['minimizeAlgorithm']

    # determine the best hyperparameters
    boostParamGrid = trainParam['boostParamGrid']
    minimizeParam = trainParam['minimizeParam']
    bestOptList = optimizeFun(
        lambda optionList: evaluateBoostParam(
            optionList, inData, outData, weights),
        boostParamGrid,
        minimizeParam
    )
    msg = formatOptions(bestOptList[0])

    # determine optimal cutoff
    predOutData = np.zeros((len(outData),))
    foldCount = trainParam['foldCount']
    folds = jrboost.stratifiedRandomFolds(outData, foldCount)
    for trainSamples, testSamples in folds:

        trainInData = jrboost.selectRows(inData, trainSamples)
        trainOutData = outData[trainSamples]
        trainWeights = weights[trainSamples]
        trainer = jrboost.BoostTrainer(trainInData, trainOutData, trainWeights)
        predictor = jrboost.EnsemblePredictor(jrboost.parallelTrain(trainer, bestOptList))

        testInData = jrboost.selectRows(inData, testSamples)
        predOutData[testSamples] = predictor.predict(testInData)

    estCutoff, _ = optimalCutoff(outData, predOutData, weights)

    # build predictor
    trainer = jrboost.BoostTrainer(inData, outData, weights)
    predictor = jrboost.EnsemblePredictor(jrboost.parallelTrain(trainer, bestOptList))

    return predictor, estCutoff, msg


def evaluateBoostParam(boostParamList, inData, outData, weights):

    foldCount = trainParam['foldCount']
    lossFun = trainParam['lossFun']

    boostParamCount = len(boostParamList)
    loss = np.zeros((boostParamCount,))
    folds = jrboost.stratifiedRandomFolds(outData, foldCount)
    for trainSamples, testSamples in folds:

        print('.', end = '', flush = True)

        trainInData = jrboost.selectRows(inData, trainSamples)
        trainOutData = outData[trainSamples]
        trainWeights = weights[trainSamples]
        trainer = jrboost.BoostTrainer(trainInData, trainOutData, trainWeights)

        testInData = jrboost.selectRows(inData, testSamples)
        testOutData = outData[testSamples]
        testWeights = weights[testSamples]

        loss += jrboost.parallelTrainAndEvalWeighted(trainer, boostParamList, testInData, testOutData, testWeights, lossFun)

    print()

    return loss


#-----------------------------------------------------------------------------------------------

def loadData(trainFrac = None):

    dataFilePath = 'C:/Data/Higgs/atlas-higgs-challenge-2014-v2.csv'
    dataFrame = pd.read_csv(dataFilePath, sep = ',', index_col = 0)

    trainSamples = dataFrame.index[dataFrame['KaggleSet'] == 't']
    if trainFrac is not None and trainFrac != 1.0:
        trainSampleCount = len(trainSamples)
        trainSamples = pd.Index(random.sample(
            trainSamples.tolist(),
            round(trainFrac * trainSampleCount)
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
    usr = opt.get('usedSampleRatio', 1.0)
    uvr = opt.get('usedVariableRatio', 1.0)
    mns = opt.get('minNodeSize', 1)
    md = opt.get('maxDepth', 1)
    pf = opt.get('pruneFactor', 0.0)
    return f'eta = {eta}  usr = {usr}  uvr = {uvr}  mns = {mns}  md = {md}  pf = {pf}'

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

#-----------------------------------------------------------------------------------------------------------------------

def _amsScoreImpl(s, b):

    b_r = 10.0
    return math.sqrt( 2.0 * (
		(s + b + b_r) 
        * math.log (1.0 + s / (b + b_r)) 
        - s
	))


def amsScore(outData, predData, weights, cutoff):
    truePos = np.sum(outData * (predData >= cutoff) * weights)
    trueNeg = np.sum((1 - outData) * (predData >= cutoff) * weights)
    return _amsScoreImpl(truePos, trueNeg)


def optimalCutoff(outData, predData, weights):

    truePos = 0.0
    falsePos = 0.0
    a = sorted(list(zip(outData, predData, weights)), key = lambda x: -x[1])

    bestScore = _amsScoreImpl(truePos, falsePos)
    bestI = -1

    for i, (outValue, _, weight) in enumerate(a):

        if outValue:
            truePos += weight
        else:
            falsePos += weight

        score = _amsScoreImpl(truePos, falsePos)
        if score <= bestScore: continue

        if i != len(a) - 1 and a[i][1] == a[i + 1][1]: continue

        bestScore = score
        bestI = i
   
    if bestI == -1:
        bestCutoff = 1.0
    elif bestI == len(a) - 1:
        bestCutoff = 0.0
    else:
        bestCutoff = (a[bestI][1] + a[bestI + 1][1]) / 2.0 

    return bestCutoff, bestScore

#---------------------------------------------------------------------------------------------

while True:
    main()


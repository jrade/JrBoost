#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import itertools,  os, time
import numpy as np
import pandas as pd
import jrboost

#-----------------------------------------------------------------------------------------------------------------------

validationParam = {
    'threadCount': os.cpu_count() // 2,
    'parallelTree': False,
    'outerFoldCount': 10,
}

trainParam = {
    'minimizeAlgorithm': jrboost.minimizePopulation,
    'repetionCount': 1,
    'innerFoldCount': 3,
    'lossFun': jrboost.logLoss,

    'boostParamGrid': {
        'iterationCount': [300],
        'eta':  [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
        'usedSampleRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'usedVariableRatio': [0.5],
        'minNodeSize': [1, 2, 3],
        'maxDepth': [1, 2, 3, 4],
        'minRelSampleWeight': [0.01],

        #'saveMemory': [True],
        #'isStratified': [False],
        'selectVariablesByLevel': [True],
    },

    'minimizeParam' : {
        'populationCount': 100,
        'survivorCount': 50,
        'cycleCount': 2,
        'bestCount': 10,
    }
}

#result (average of 100 runs)
#
#Species          Iris-setosa  Iris-versicolor  Iris-virginica
#Species
#Iris-setosa             50.0             0.00            0.00
#Iris-versicolor          0.0            46.85            3.15
#Iris-virginica           0.0             3.22           46.78

#-----------------------------------------------------------------------------------------------------------------------

def validate():

    print(f'validation: {validationParam}\n')
    print(f'train: {trainParam}\n')

    if 'threadCount' in validationParam: jrboost.setThreadCount(validationParam['threadCount'])
    if 'parallelTree' in validationParam: jrboost.setParallelTree(validationParam['parallelTree'])

    outerFoldCount = validationParam['outerFoldCount']

    inDataFrame, outDataSeries = loadData()
    outDataFrame = jrboost.oneHotEncode(outDataSeries)

    samples = outDataFrame.index
    variables = inDataFrame.columns
    labels = outDataFrame.columns
    print(f'{len(samples)} samples, {len(variables)} variables\n')

    confusionFrame = pd.DataFrame(index = labels, columns = labels, data = 0)

    inData = inDataFrame.to_numpy(dtype = np.float32)
    #for i in range(100):  
    for i in itertools.count():


        print(f'-------------------- {i} --------------------\n')

        t = -time.time()
        jrboost.PROFILE.START() 
        predOutDataFrame = pd.DataFrame(index = samples, columns = labels, dtype = np.float64)

        for label in labels:
            print(label)

            outData = outDataFrame[label].to_numpy(dtype = np.uint64)
            predOutData = np.empty((len(samples),))

            folds = jrboost.stratifiedRandomFolds(outData, outerFoldCount)
            for trainSamples, testSamples in folds:

                trainInData = inData[trainSamples, :]
                trainOutData = outData[trainSamples]
                predictor = train(trainInData, trainOutData)

                testInData = inData[testSamples, :]
                predOutData[testSamples] = predictor.predict(testInData)
           
            predOutDataFrame[label] = predOutData
            print()

        print()
        t += time.time()
        print(jrboost.PROFILE.STOP())
        print(f'{t:.2f}s')
        print()

        predOutDataSeries = predOutDataFrame.idxmax(axis = 1)
        for sample in samples:
            confusionFrame.loc[outDataSeries[sample], predOutDataSeries[sample]] += 1

        print(confusionFrame / (i + 1))
        print()

#-----------------------------------------------------------------------------------------------------------------------

def loadData():
    dataPath = '../Data/Iris/Iris.csv'
    dataFrame = pd.read_csv(dataPath, sep = ',', index_col = 0)
    outDataSeries = dataFrame['Species']
    inDataFrame = dataFrame.drop(['Species'], axis = 1)
    return inDataFrame, outDataSeries


def train(inData, outData):

    minimizeAlgorithm = trainParam['minimizeAlgorithm']
    repetionCount = trainParam['repetionCount']
    boostParamGrid = trainParam['boostParamGrid']
    minimizeParam = trainParam['minimizeParam']

    bestBoostParamList = []
    for _ in range(repetionCount):
        bestBoostParamList += minimizeAlgorithm(
            lambda boostParamList: evaluateBoostParam(boostParamList, inData, outData),
            boostParamGrid,
            minimizeParam
        )                 
    print(formatBoostParam(bestBoostParamList[0]))

    trainer = jrboost.BoostTrainer(inData, outData)
    predictorList = [trainer.train(boostParam) for boostParam in bestBoostParamList]
    predictor = jrboost.EnsemblePredictor(predictorList)
    return predictor


def evaluateBoostParam(boostParamList, inData, outData):

    innerFoldCount = trainParam['innerFoldCount']
    lossFun = trainParam['lossFun']

    boostParamCount = len(boostParamList)
    loss = np.zeros((boostParamCount,))
    folds = jrboost.stratifiedRandomFolds(outData, innerFoldCount)
    for trainSamples, testSamples in folds:

        trainInData = inData[trainSamples, :]
        trainOutData = outData[trainSamples]
        trainer = jrboost.BoostTrainer(trainInData, trainOutData)

        testInData = inData[testSamples, :]
        testOutData = outData[testSamples]
        loss += jrboost.parallelTrainAndEval(trainer, boostParamList, testInData, testOutData, lossFun)

    return loss

#-----------------------------------------------------------------------------------------------------------------------

def formatBoostParam(boostParam):
    eta  = boostParam['eta']
    md = boostParam.get('maxDepth', 1)
    usr = boostParam['usedSampleRatio']
    uvr = boostParam['usedVariableRatio']
    mns = boostParam['minNodeSize']
    return f'eta = {eta:.4f}  md = {md}  usr = {usr:.1f}  uvr = {uvr:.2f}  mns = {mns}'

#-----------------------------------------------------------------------------------------------------------------------

validate()

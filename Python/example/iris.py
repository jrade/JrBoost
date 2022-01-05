#  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import itertools, pickle, os, time
import numpy as np
import pandas as pd
import jrboost

#-----------------------------------------------------------------------------------------------------------------------

validationParam = {
    'threadCount': os.cpu_count() // 2,
    'parallelTree': False,
    'foldCount': 10,
}

trainParam = {
    'minimizeAlgorithm': jrboost.minimizePopulation,
    'repetionCount': 1,
    'foldCount': 3,
    'targetLossFun': jrboost.logLoss,

    'boostParamGrid': {
        'iterationCount': [400], #[100, 150, 200, 300, 500, 750, 1000],
        'eta':  [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
        'cycle':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'usedSampleRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'usedVariableRatio': [0.5],
        'minNodeSize': [1, 2, 3],
        'maxTreeDepth': [1, 2, 3, 4],
        'minRelSampleWeight': [0.01],

        #'saveMemory': [True],
        #'stratifiedSamples': [False],
        'selectVariablesByLevel': [True],
        #'fastExp': [False],
    },

    'minimizeParam' : {
        'populationCount': 100,
        'survivorCount': 50,
        'cycleCount': 2,
        'bestCount': 10,
    }
}

#-----------------------------------------------------------------------------------------------------------------------

def main():

    print(f'validation: {jrboost.formatParam(validationParam)}\n')
    print(f'train: {jrboost.formatParam(trainParam)}\n')

    if 'threadCount' in validationParam: jrboost.setThreadCount(validationParam['threadCount'])
    if 'parallelTree' in validationParam: jrboost.setParallelTree(validationParam['parallelTree'])

    outerFoldCount = validationParam['foldCount']

    inDataFrame, outDataSeries = loadData()
    outDataFrame = jrboost.oneHotEncode(outDataSeries)

    samples = outDataFrame.index
    variables = inDataFrame.columns
    labels = outDataFrame.columns
    print(f'{len(samples)} samples, {len(variables)} variables\n')

    confusionFrame = pd.DataFrame(index = labels, columns = labels, data = 0)

    inData = inDataFrame.to_numpy(dtype = np.float32)
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

                bestBoostParams = jrboost.optimizeHyperParam(trainInData, trainOutData, trainParam)
                trainer = jrboost.BoostTrainer(trainInData, trainOutData)
                predictor = jrboost.Predictor.createEnsemble(jrboost.parallelTrain(trainer, bestBoostParams))
                print(formatBoostParam(jrboost.medianBoostParam(bestBoostParams)))

                #predictor.save('foo.bin')
                #predictor = jrboost.Predictor.load('foo.bin')

                testInData = inData[testSamples, :]
                predOutData[testSamples] = predictor.predict(testInData)
           
            predOutDataFrame[label] = predOutData
            print()

        print()
        t += time.time()
        print(jrboost.PROFILE.STOP())
        print(f'{t:.2f}s\n')

        predOutDataSeries = predOutDataFrame.idxmax(axis = 1)
        for sample in samples:
            confusionFrame.loc[outDataSeries[sample], predOutDataSeries[sample]] += 1

        print((confusionFrame / (i + 1)).to_string(float_format = lambda x: f'{x:.2f}') + '\n')

#-----------------------------------------------------------------------------------------------------------------------

def loadData():
    dataPath = '../Data/Iris/Iris.csv'
    dataFrame = pd.read_csv(dataPath, sep = ',', index_col = 0)
    outDataSeries = dataFrame['Species']
    inDataFrame = dataFrame.drop(['Species'], axis = 1)
    return inDataFrame, outDataSeries


def formatBoostParam(boostParam):
    ic  = boostParam['iterationCount']
    eta  = boostParam['eta']
    cc  = boostParam['cycle']
    md = boostParam.get('maxTreeDepth', 1)
    usr = boostParam['usedSampleRatio']
    mns = boostParam['minNodeSize']
    return f'  ic = {ic}  eta = {eta:.4f}  cc = {cc:.1f}  md = {md}  usr = {usr:.1f}  mns = {mns}'

#-----------------------------------------------------------------------------------------------------------------------

main()


#result (average of 100 runs)
#
#Species          Iris-setosa  Iris-versicolor  Iris-virginica
#Species
#Iris-setosa             50.0             0.00            0.00
#Iris-versicolor          0.0            46.85            3.15
#Iris-virginica           0.0             3.22           46.78


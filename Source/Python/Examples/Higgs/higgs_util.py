#  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import math, os
import numpy as np
import pandas as pd
import jrboost

dataDirPath = 'C:/Data/Higgs'
resultDirPath = '../../../../Higgs Result'

#-----------------------------------------------------------------------------------------------------------------------

def loadTrainData():

    print('Loading training data')

    trainDataFilePath = dataDirPath + '/training.csv'

    trainDataFrame = pd.read_csv(trainDataFilePath, sep = ',', index_col = 0)
    trainDataFrame.index.name = 'EventId'

    trainOutDataSeries = trainDataFrame['Label'].map({'b': 0, 's': 1})
    trainWeightSeries = trainDataFrame['Weight']
    trainInDataFrame = trainDataFrame.drop(['Label', 'Weight'], axis = 1)

    return trainInDataFrame, trainOutDataSeries, trainWeightSeries


def splitData(inData, outData,  weights, ratio):
    samples1, samples2 = jrboost.stratifiedRandomSplit(outData, ratio)
    return inData[samples1, :], outData[samples1], weights[samples1], inData[samples2, :], outData[samples2], weights[samples2]

#-----------------------------------------------------------------------------------------------------------------------

def saveResult(predictor, threshold, fileName):

    print('Loading test data')
    testDataFilePath = dataDirPath + '/test.csv'
    testInDataFrame = pd.read_csv(testDataFilePath, sep = ',', index_col = 0)
    testInDataFrame.index.name = 'EventId'

    print('Testing predictor')
    testInData = testInDataFrame.to_numpy(dtype = np.float32)
    testPredData = predictor.predict(testInData)
    testPredDataSeries = pd.Series(
        index = testInDataFrame.index,
        data = testPredData
    )

    print('Saving result')
    resultDataFrame = pd.DataFrame({
        'RankOrder': testPredDataSeries.rank(method = 'first', ascending = False).astype(int),
        'Class': (testPredDataSeries >= threshold).map({False: 'b', True: 's'}) 
    })
    os.makedirs(resultDirPath, exist_ok = True)
    resultFilePath = resultDirPath + '/' + fileName
    resultDataFrame.to_csv(resultFilePath, sep = ',')

    print()
    print(f'The result has been saved to "{os.path.abspath(resultFilePath)}".')
    print(f'It can be uploaded to https://www.kaggle.com/c/higgs-boson/submit and scored.')
    print()


#-----------------------------------------------------------------------------------------------------------------------

def _amsScore(s, b):
    b_r = 10.0
    return math.sqrt( 2.0 * (
        (s + b + b_r) 
        * math.log (1.0 + s / (b + b_r)) 
        - s
    ))


def optimalTheshold(outData, predData, weights):

    # test all possible thresholds

    a = sorted(list(zip(outData, predData, weights)), key = lambda x: -x[1])

    truePos = 0.0
    falsePos = 0.0

    bestScore = _amsScore(truePos, falsePos)
    bestI = -1

    for i, (outValue, _, weight) in enumerate(a):

        if outValue:
            truePos += weight
        else:
            falsePos += weight

        score = _amsScore(truePos, falsePos)
        if score <= bestScore: continue

        bestScore = score
        bestI = i

    assert bestI != -1
    assert bestI != len(a) - 1
    assert a[bestI][1] != a[bestI + 1][1]     # How to handle this?     

    bestTheshold = (a[bestI][1] + a[bestI + 1][1]) / 2.0 

    return bestTheshold, bestScore

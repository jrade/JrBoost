
import sys
sys.path += ['.', '../..']

import os
import numpy as np
import pandas as pd
import jrboost


dataPath = 'Data/Agaricus.csv'


def test():

    print('Agaricus test -------------------------\n')

    threadCount = os.cpu_count() // 2
    jrboost.setThreadCount(threadCount)

    inDataFrame, outDataSeries = loadData(dataPath)
    assert (inDataFrame.index == outDataSeries.index).all()

    samples = inDataFrame.index
    variables = inDataFrame.columns
    sampleCount = len(samples)
    variableCount = len(variables)
    print(f'{sampleCount} samples, {variableCount} variables')
    print('labels: 0 = edible, 1 = poisonous\n')

    inData = inDataFrame.to_numpy()
    outData = outDataSeries.to_numpy()
    prob = np.empty((len(samples),))

    folds = jrboost.stratifiedRandomFolds(outData, 20)
    for trainSamples, testSamples in folds:
        trainInData = inData[trainSamples, :]
        trainOutData = outData[trainSamples]
        testInData = inData[testSamples, :]
        testOutData = outData[testSamples]

        trainer = jrboost.BoostTrainer(trainInData, trainOutData)
        predictor = trainer.train({})
        testProb = predictor.predict(testInData)
        prob[testSamples] = testProb

    predOutData = (prob > 0.5).astype(int)

    errorCount = (predOutData != outData).sum()
    print(f'error count = {errorCount}')
    accuracy = (sampleCount - errorCount) / sampleCount;
    print(f'accuracy = {accuracy}\n')

    confusionFrame = pd.DataFrame(data = np.zeros((2,2), dtype = int))
    for i in range(sampleCount):
        confusionFrame.loc[outData[i], predOutData[i]] += 1
    print(confusionFrame.to_string(float_format = lambda x: f'{x:.2f}') + '\n')

    ok = accuracy > 0.99
    if ok:
        print('Test accuracy > 0.99 passed\n\n')
    else:
        print('Test accuracy > 0.99 failed\n\n')
    return ok




def charToInt1(c):
    if c == 'e': return 0
    if c == 'p': return 1
    assert false

def charToInt2(c):
    return ord(c) - ord('a')

def loadData(dataPath):
    dataFrame = pd.read_csv(dataPath, sep = ",", index_col = None, keep_default_na = False)
    inDataFrame = pd.DataFrame(
        data = dataFrame.drop('class', axis = 1).applymap(charToInt2),
        columns = dataFrame.columns.drop('class'),
        dtype = np.float32
    )
    outDataSeries = pd.Series(
        data = dataFrame.loc[:, 'class'].map(charToInt1),
        dtype = np.uint8
    )
    return inDataFrame, outDataSeries


def loadSeries(outDataPath):
    series = pd.read_csv(outDataPath, sep = "\t", index_col = 0, keep_default_na = False).squeeze()
    return series


if (__name__ == '__main__'):
    test()

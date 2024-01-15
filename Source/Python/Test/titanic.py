import sys
sys.path += ['.', '../..']

import os
import numpy as np
import pandas as pd
import jrboost


inDataPath = 'Data/Titanic/InData.tsv'
outDataPath = 'Data/Titanic/OutData.tsv'

def test():

    print('Titanic test --------------------------\n')

    threadCount = os.cpu_count() // 2
    jrboost.setThreadCount(threadCount)

    inDataFrame = loadInDataFrame(inDataPath)
    outDataSeries = loadSeries(outDataPath)
    assert (inDataFrame.index == outDataSeries.index).all()

    samples = inDataFrame.index
    variables = inDataFrame.columns
    sampleCount = len(samples)
    variableCount = len(variables)

    print(f'{sampleCount} samples, {variableCount} variables')
    print('labels: 0 = died, 1 = survived\n')

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
        #predictor = trainer.train({})
        predictor = trainer.train({'iterationCount': 300, 'eta': 0.01})
        testProb = predictor.predict(testInData)
        prob[testSamples] = testProb

    predOutData = (prob > 0.5).astype(int)

    errorCount = (outData != predOutData).sum()
    print(f'error count = {errorCount}')
    accuracy = (sampleCount - errorCount) / sampleCount;
    print(f'accuracy = {accuracy}\n')

    confusionFrame = pd.DataFrame(data = np.zeros((2,2), dtype = int))
    for i in range(sampleCount):
        confusionFrame.loc[outData[i], predOutData[i]] += 1
    print(confusionFrame.to_string(float_format = lambda x: f'{x:.2f}') + '\n')

    ok = accuracy > 0.80
    if ok:
        print('Test accuracy > 0.80 passed\n\n')
    else:
        print('Test accuracy > 0.80 failed\n\n')
    return ok



def loadInDataFrame(inDataPath):
    inDataFrame = pd.read_csv(inDataPath, sep = "\t", index_col = 0, keep_default_na = False)
    inDataFrame = pd.DataFrame(
        data = np.ascontiguousarray(inDataFrame.to_numpy(), dtype = np.float32),
        index = inDataFrame.index,
        columns = inDataFrame.columns,
    )
    return inDataFrame


def loadSeries(outDataPath):
    series = pd.read_csv(outDataPath, sep = "\t", index_col = 0, keep_default_na = False).squeeze()
    return series


if (__name__ == '__main__'):
    test()

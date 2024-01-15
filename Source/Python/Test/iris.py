#  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import sys
sys.path += ['.', '../..']

import numpy as np
import pandas as pd
import jrboost


#-----------------------------------------------------------------------------------------------------------------------

def test():

    print('Iris test -----------------------------\n')

    foldCount = 20

    inDataFrame, outDataSeries = loadData()
    outDataFrame = jrboost.oneHotEncode(outDataSeries)

    samples = outDataFrame.index
    variables = inDataFrame.columns
    sampleCount = len(samples)
    labels = outDataFrame.columns
    variableCount = len(variables)
    labelCount = len(labels)
    print(f'{sampleCount} samples, {variableCount} variables, {labelCount} labels\n')

    inData = inDataFrame.to_numpy(dtype = np.float32)

    predOutDataFrame = pd.DataFrame(index = samples, columns = labels, dtype = np.float64)

    for label in labels:
        outData = outDataFrame[label].to_numpy(dtype = np.uint8)
        predOutData = np.empty((len(samples),))

        folds = jrboost.stratifiedRandomFolds(outData, foldCount)
        for trainSamples, testSamples in folds:

            trainInData = inData[trainSamples, :]
            trainOutData = outData[trainSamples]

            trainer = jrboost.BoostTrainer(trainInData, trainOutData)
            predictor = trainer.train({'iterationCount': 300, 'eta': 0.01, 'usedSampleRatio': 0.3})

            testInData = inData[testSamples, :]
            predOutData[testSamples] = predictor.predict(testInData)
           
        predOutDataFrame[label] = predOutData

    predOutDataSeries = predOutDataFrame.idxmax(axis = 1)

    errorCount = (predOutDataSeries != outDataSeries).sum()
    print(f'error count = {errorCount}')
    accuracy = (sampleCount - errorCount) / sampleCount;
    print(f'accuracy = {accuracy}\n')



    confusionFrame = pd.DataFrame(index = labels, columns = labels, data = 0)
    for sample in samples:
        confusionFrame.loc[outDataSeries[sample], predOutDataSeries[sample]] += 1

    print(confusionFrame.to_string(float_format = lambda x: f'{x:.2f}') + '\n')

    ok = accuracy > 0.90
    if ok:
        print('Test accuracy > 0.90 passed\n\n')
    else:
        print('Test accuracy > 0.90 failed\n\n')
    return ok


#-----------------------------------------------------------------------------------------------------------------------

def loadData():
    dataFilePath = 'Data/Iris.csv'
    dataFrame = pd.read_csv(dataFilePath, sep = ',', index_col = 0)
    outDataSeries = dataFrame['Species']
    inDataFrame = dataFrame.drop(['Species'], axis = 1)
    return inDataFrame, outDataSeries

#-----------------------------------------------------------------------------------------------------------------------

if (__name__ == '__main__'):
    test()

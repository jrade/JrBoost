import itertools, random, sys, time
import numpy as np
import pandas as pd
import util
import jrboost

PROFILE = jrboost.PROFILE

#-----------------------------------------------------------------------------------------------------------------------

def main():

    threadCount = 4
    jrboost.setNumThreads(threadCount)
    jrboost.setProfile(True)
    print(f'{threadCount} threads\n')

    outerFoldCount = 20
    innerFoldCount = 5
    bestOptionCount = 5
    lossFun = jrboost.linLoss

    inDataFrame, outDataSeries = loadData()
    outDataFrame = util.oneHotEncode(outDataSeries)

    samples = outDataFrame.index
    variables = inDataFrame.columns
    labels = outDataFrame.columns
    print(f'{len(samples)} samples, {len(variables)} variables')
    print(f'labels: {list(labels)}')
    print()

    inData = inDataFrame.to_numpy(dtype = np.float32)
    for i in itertools.count():

        print(f'-------------------- {i} --------------------\n')
        t = -time.time()
        PROFILE.PUSH(PROFILE.MAIN)
        predOutDataFrame = pd.DataFrame(index = samples, columns = labels, dtype = np.float64)

        for label in labels:
            print(label + ' ', end = '', flush = True)
            outData = outDataFrame[label].to_numpy(dtype = np.uint64)

            predOutData = np.empty((len(samples),))
            folds = util.stratifiedRandomFolds(outData, outerFoldCount)
            for trainSamples, testSamples in folds:
                print('.', end = '', flush = True)
                sortedOpts = optimizeHyperParams(inData, outData, trainSamples, innerFoldCount, lossFun)
                bestOpts = sortedOpts[:bestOptionCount]
                #for opt in bestOpts:
                #    print(formatOptions(opt))
                predOutData[testSamples] = util.trainAndPredictInternal(inData, outData, trainSamples, testSamples, bestOpts)

            print()
            predOutDataFrame[label] = predOutData

        print()
        predOutDataSeries = predOutDataFrame.idxmax(axis = 1)
        confusionFrame = pd.DataFrame(index = labels, columns = labels, data = 0)
        for sample in samples:
            confusionFrame.loc[outDataSeries[sample], predOutDataSeries[sample]] += 1

        PROFILE.POP()
        t += time.time()
        PROFILE.PRINT()
        print()
        print(confusionFrame)
        print()
        print(util.formatTime(t))
        print()

#-----------------------------------------------------------------------------------------------------------------------

def loadData():

    dataPath = util.findPath('/Data/Iris/Iris.csv')
    dataFrame = pd.read_csv(dataPath, sep = ',', index_col = 0)
    outDataSeries = dataFrame['Species']
    inDataFrame = dataFrame.drop(['Species'], axis = 1)

    return inDataFrame, outDataSeries

#-----------------------------------------------------------------------------------------------------------------------

def formatOptions(opt):
    ic = opt.iterationCount
    eta  = opt.eta
    usr = opt.usedSampleRatio
    uvr = opt.usedVariableRatio
    return f'ic = {ic}  eta = {eta}  usr = {usr:.1f}  uvr = {uvr:.1f}'


def optimizeHyperParams(inData, outData, samples, foldCount, lossFun):

    optionsList = []
    for eta in [0.2, 0.5, 1.0, 2.0]:
        for usr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for uvr in [0.1, 0.3, 0.5, 0.7, 0.9]:
                opt = jrboost.BoostOptions()
                opt.iterationCount = 1000
                opt.eta = eta
                opt.usedSampleRatio = usr
                opt.usedVariableRatio = uvr
                opt.minSampleWeight = 0.001
                optionsList.append(opt)

    optionsCount = len(optionsList)
    loss = np.zeros((optionsCount,))
    folds = util.stratifiedRandomFolds(outData, foldCount, samples)
    for trainSamples, testSamples in folds:
        loss += util.trainAndEvalInternal(inData, outData, trainSamples, testSamples, optionsList, lossFun)
            
    sortedOptionsList = [optionsList[i] for i in np.argsort(loss)]
    return sortedOptionsList

#-----------------------------------------------------------------------------------------------------------------------

main()

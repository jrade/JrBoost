import itertools, random, sys, time
import numpy as np
import pandas as pd
import util, optimize_grid, optimize_dynamic
import jrboost

PROFILE = jrboost.PROFILE

#-----------------------------------------------------------------------------------------------------------------------

cvParam = {
    'threadCount': 4,
    'profile': False,

    'optimizeFun': optimize_grid.optimize,
    'lossFun': jrboost.negAuc, #jrboost.linLoss,

    'boostParamValues': {
        'iterationCount': [1000],
        'eta':  [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
        'usedSampleRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'usedVariableRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'minNodeSize': [1, 2, 3, 4, 6, 8, 10],
        'minSampleWeight': [0.001],
    },

    'outerFoldCount': 5,
    'innerFoldCount': 5,
    'bestOptionCount': 1,
    'bagSize': 1,

    'populationCount': 100,
    'survivorCount': 50,
    'cycleCount': 10,
}

#-----------------------------------------------------------------------------------------------------------------------

def main():

    print(cvParam)
    print()

    threadCount = cvParam['threadCount']
    profile = cvParam['profile']
    outerFoldCount = cvParam['outerFoldCount']
    innerFoldCount = cvParam['innerFoldCount']
    optimizeFun = cvParam['optimizeFun']
    lossFun = cvParam['lossFun']

    jrboost.setProfile(profile)
    jrboost.setNumThreads(threadCount)
    print(f'{threadCount} threads\n')

    inDataFrame, outDataSeries = loadData()
    outDataFrame = util.oneHotEncode(outDataSeries)
    samples = outDataFrame.index
    variables = inDataFrame.columns
    labels = outDataFrame.columns
    print(f'{len(samples)} samples, {len(variables)} variables')
    print(f'labels: {list(labels)}\n')

    inData = inDataFrame.to_numpy(dtype = np.float32)
    for i in itertools.count():

        print(f'-------------------- {i} --------------------\n')
        t = -time.time()
        PROFILE.PUSH(PROFILE.MAIN)
        predOutDataFrame = pd.DataFrame(index = samples, columns = labels, dtype = np.float64)

        for label in labels:
            print(label)
            outData = outDataFrame[label].to_numpy(dtype = np.uint64)

            predOutData = np.empty((len(samples),))
            folds = util.stratifiedRandomFolds(outData, outerFoldCount)
            for trainSamples, testSamples in folds:
                bestOptList = optimizeFun(
                    cvParam,
                    lambda optionList: 
                        util.trainAndEvalInternal(inData, outData, trainSamples, innerFoldCount, optionList, lossFun)
                )                 
                print(formatOptions(bestOptList[0]))
                predOutData[testSamples] = util.trainAndPredictInternal(inData, outData, trainSamples, testSamples, bestOptList)

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


def formatOptions(opt):
    ic = opt.iterationCount
    eta  = opt.eta
    usr = opt.usedSampleRatio
    uvr = opt.usedVariableRatio
    mns = opt.minNodeSize
    return f'ic = {ic}  eta = {eta}  usr = {usr:.1f}  uvr = {uvr:.1f}  mns = {mns:2}'

#-----------------------------------------------------------------------------------------------------------------------

main()

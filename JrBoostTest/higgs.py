import random, time
import numpy as np
import pandas as pd
import util, optimize_grid, optimize_dynamic
import jrboost

PROFILE = jrboost.PROFILE

#-----------------------------------------------------------------------------------------------------------------------

cvParam = {
    'threadCount': 4,
    'profile': True,
    'trainFraction' : 0.001,

    'optimizeFun': optimize_dynamic.optimize,
    'lossFun': jrboost.negAuc,

    'boostParamValues': {
        'iterationCount': [1000],
        'eta':  [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
        'usedSampleRatio': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        'usedVariableRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'minNodeSize': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        'minSampleWeight': [0.000001],
    },

    'bestOptionCount': 10,
    'bagSize': 1,

    'populationCount': 100,
    'survivorCount': 50,
    'cycleCount': 10,
}

print(cvParam)
print()


#-----------------------------------------------------------------------------------------------------------------------

def main():

    threadCount = cvParam['threadCount']
    profile = cvParam['profile']
    trainFraction = cvParam.get('trainFraction', None)
    optimizeFun = cvParam['optimizeFun']
    lossFun = cvParam['lossFun']

    jrboost.setProfile(profile)
    jrboost.setNumThreads(threadCount)
    print(f'{threadCount} threads\n')

    (trainInDataFrame, trainOutDataSeries, trainWeightSeries,
        testInDataFrame, testOutDataSeries,
        validationInDataFrame, validationOutDataSeries) = loadData(trainFrac = trainFraction)

    trainSamples = trainInDataFrame.index
    testSamples = testInDataFrame.index
    validationSamples = validationInDataFrame.index

    trainInData = trainInDataFrame.to_numpy(dtype = np.float32)
    trainOutData = trainOutDataSeries.to_numpy(dtype = np.uint64)
    trainWeights = trainWeightSeries.to_numpy(dtype = np.float32)
    testInData = testInDataFrame.to_numpy(dtype = np.float32)
    testOutData = testOutDataSeries.to_numpy(dtype = np.uint64)
    validationInData = validationInDataFrame.to_numpy(dtype = np.float32)
    validationOutData = validationOutDataSeries.to_numpy(dtype = np.uint64)

    # make sure that total weight is the same for background and signal events
    trainWeights[trainOutData == 0] /= np.sum(trainWeights[trainOutData == 0])
    trainWeights[trainOutData == 1] /= np.sum(trainWeights[trainOutData == 1])
    # normalize so largest weight is 1
    trainWeights /= np.max(trainWeights)

    print(f'{len(trainSamples)} train samples, {len(testSamples)} test samples, {len(validationSamples)} validation samples\n')

    #...............................................................................

    t = -time.time()
    PROFILE.PUSH(PROFILE.MAIN)
    #predOutDataFrame = pd.DataFrame(index = testSamples, columns = labels, dtype = np.uint64)

    bestOptList = optimizeFun(
        cvParam,
        lambda optionList: trainAndEval(
            trainInData, trainOutData, trainWeights, testInData, testOutData, optionList, lossFun)
    )
                                                      
    print(formatOptions(bestOptList[0]))

    predOutData = trainAndPredict(trainInData, trainOutData, validationInData, bestOptList)

    estSignalRatio = sum(trainOutData) / len(trainOutData)
    estCutoff = np.quantile(predOutData, 1.0 - estSignalRatio)

    print(f'threshold = {estCutoff}')
   
    PROFILE.POP()
    t += time.time()
    PROFILE.PRINT()
    print(util.formatTime(t))
    print()
    print()

    submission = pd.DataFrame(
        index = validationSamples,
        data = {
            'P': predOutData,
            'RankOrder': ranks(predOutData) + 1,
            'Class': np.where(predOutData > estCutoff, 's', 'b')
        }
    )
    submission.to_csv('Higgs Submission.csv', sep = ',')

    print("Done!")

#-----------------------------------------------------------------------------------------------------------------------

def ranks(data):
    temp = data.argsort()
    ranks1 = np.empty_like(temp)
    ranks1[temp] = np.arange(len(temp))
    return ranks1


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
    weightSeries = dataFrame['Weight']
    inDataFrame = dataFrame.drop(['Label', 'Weight', 'KaggleSet'], axis = 1)

    trainInDataFrame = inDataFrame.loc[trainSamples, :]
    trainOutDataSeries = outDataSeries[trainSamples]
    trainWeightSeries = weightSeries[trainSamples]
    testInDataFrame = inDataFrame.loc[testSamples, :]
    testOutDataSeries = outDataSeries[testSamples]
    validationInDataFrame = inDataFrame.loc[validationSamples, :]
    validationOutDataSeries = outDataSeries[validationSamples]

    return (
        trainInDataFrame, trainOutDataSeries, trainWeightSeries,
        testInDataFrame, testOutDataSeries,
        validationInDataFrame, validationOutDataSeries
    )
    

def formatOptions(opt):
    eta = opt.eta
    usr = opt.usedSampleRatio
    uvr = opt.usedVariableRatio
    mns = opt.minNodeSize
    return f'eta = {eta:.2f}  usr = {usr:.1f}  uvr = {uvr:.1f}  mns = {mns:2}'


def formatScore(score, precision = 4):
    return '(' + ', '.join((f'{x:.{precision}f}' for x in score)) + ')'


def trainAndEval(
    trainInData, trainOutData, trainWeights,
    testInData, testOutData,
    optionList, lossFun
):
    optionCount = len(optionList)
    loss = np.zeros((optionCount,))

    trainer = jrboost.BoostTrainer(trainInData, trainOutData, trainWeights)
    loss =  trainer.trainAndEval(testInData, testOutData, optionList, lossFun)
    return loss


def trainAndPredict(trainInData, trainOutData, validationInData, bestOptList):

    predOutDataList = []
    trainer = jrboost.BoostTrainer(trainInData, trainOutData)
    for opt in bestOptList:
        predictor = trainer.train(opt)
        predOutDataList.append(predictor.predict(validationInData))
    predOutData = np.median(np.array(predOutDataList), axis = 0)
    return predOutData

#-----------------------------------------------------------------------------------------------------------------------


main()

#  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import datetime, os
import numpy as np
import jrboost
import higgs_util as util

#-----------------------------------------------------------------------------------------------------------------------

boostParam = {
    'iterationCount': 1000,
    'eta': 0.03,
    'usedSampleRatio': 0.8,
    'usedVariableRatio': 0.3,
    'maxTreeDepth': 8,
    'minNodeSize': 300,
}

#-----------------------------------------------------------------------------------------------------------------------

def main():

    threadCount = os.cpu_count() // 2
    jrboost.setThreadCount(threadCount)
    print(f'Thread count = {threadCount}\n')

    print(f'Boost parameters = {boostParam}\n')

    trainInDataFrame, trainOutDataSeries, trainWeightSeries = util.loadTrainData()

    predictor, threshold = trainPredictor(trainInDataFrame, trainOutDataSeries, trainWeightSeries)

    fileName = f'Higgs Fast Result {datetime.datetime.now().strftime("%y%m%d-%H%M%S")}.csv'
    util.saveResult(predictor, threshold, fileName)

#-----------------------------------------------------------------------------------------------------------------------

def trainPredictor(inDataFrame, outDataSeries, weightSeries):

    print('Training predictor')

    # split the data into two parts
    # the first part contains 2/3 and the second part the remaining 1/3 of the samples

    inData = inDataFrame.to_numpy(dtype = np.float32)
    outData = outDataSeries.to_numpy(dtype = np.uint8)
    weights = weightSeries.to_numpy(dtype = np.float64)

    inData1, outData1, weights1, inData2, outData2, weights2 = util.splitData(inData, outData, weights, 2/3)
    weights1 *= weights.sum() / weights1.sum()
    weights2 *= weights.sum() / weights2.sum()

    # use the first part of the data to train the predictor

    trainer = jrboost.BoostTrainer(inData1, outData1, weights = weights1)
    jrboost.PROFILE.START()
    predictor = trainer.train(boostParam)
    print()
    print(jrboost.PROFILE.STOP())

    # use the second part of the data to determine the threshold

    predOutData2 = predictor.predict(inData2)
    threshold, _ = util.optimalTheshold(outData2, predOutData2, weights2)

    return predictor, threshold
 
#-----------------------------------------------------------------------------------------------------------------------

main()

import os
os.environ['PATH'] += ';C:/Users/Rade/Anaconda3/Library/bin'

import numpy as np
import pandas as pd
import util
import jrboost

dataPath = r'C:/Users/Rade/Documents/Data Analysis/Data/Iris/Iris.csv'
dataFrame = pd.read_csv(dataPath, sep = ',', index_col = 0)
outDataSeries = dataFrame['Species']
outDataFrame = util.oneHotEncode(outDataSeries)
inDataFrame = dataFrame.drop(['Species'], axis = 1)

samples = dataFrame.index
sampleCount = len(samples)
labels = outDataFrame.columns

print(inDataFrame.head(5))
print()
print(outDataFrame.head(5))
print()

inData = inDataFrame.to_numpy()
inData = np.asfortranarray(inData, dtype = np.float32)
weights = np.full((sampleCount,), 1.0)

#---------------------------------------------------------------

baseOpt = jrboost.StumpOptions()
baseOpt.usedSampleRatio = 1
baseOpt.usedVariableRatio = 0.2
baseOpt.profile = False

opt = jrboost.BoostOptions()
opt.iterationCount = 100
opt.eta = 0.3
opt.baseOptions = baseOpt


predFrame = pd.DataFrame(index = samples, columns = labels)

for label in labels:

    outData = outDataFrame[label].to_numpy();
    trainer = jrboost.AdaBoostTrainer(inData, outData, weights)
    predictor = trainer.train(opt)
    predFrame[label] = predictor.predict(inData)


predSeries = predFrame.idxmax(axis = 1)

confusionFrame = pd.DataFrame(index = labels, columns = labels, data = 0)
for sample in samples:
    confusionFrame.loc[outDataSeries[sample], predSeries[sample]] += 1

print()
print(confusionFrame)
print()
print("Done!")



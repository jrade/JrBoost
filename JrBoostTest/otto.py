import os
os.environ['PATH'] += ';C:/Users/Rade/Anaconda3/Library/bin'

import numpy as np
import pandas as pd
import util
import jrboost

dataPath = r'C:/Users/Rade/Documents/Data Analysis/Data/Otto/train.csv'
dataFrame = pd.read_csv(dataPath, sep = ',', index_col = 0)

#dataFrame = dataFrame.sample(frac = 0.2)

outDataSeries = dataFrame['target']
outDataFrame = util.oneHotEncode(outDataSeries)
inDataFrame = dataFrame.drop(['target'], axis = 1)

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

baseOpt = jrboost.StumpOptions()
baseOpt.usedSampleRatio = 1
baseOpt.usedVariableRatio = 0.2
baseOpt.highPrecision = True
baseOpt.profile = False

opt = jrboost.AdaBoostOptions()
#opt = jrboost.LogitBoostOptions()
opt.iterationCount = 1000
opt.eta = 1.0
opt.highPrecision = True
opt.baseOptions = baseOpt

trainer = opt.createTrainer()
trainer.setInData(inData)
trainer.setWeights(weights)

predFrame = pd.DataFrame(index = samples, columns = labels)

for label in labels:

    print(label)

    outData = outDataFrame[label].to_numpy();
    trainer.setOutData(outData)
    predictor = trainer.train()
    predFrame[label] = predictor.predict(inData)

predFrame = 1.0 / (1.0 + np.exp(-predFrame))
predFrame = predFrame.divide(predFrame.sum(axis = 1), axis = 0)
predFrame = predFrame.clip(1.0e-15, 1.0)
score = -(np.log(predFrame) * outDataFrame).sum().sum() / sampleCount

print()
print(score)   # 0.5121  (1523 / 3507)

print()
print("Done!")

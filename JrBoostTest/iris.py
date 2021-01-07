import os
os.environ['PATH'] += ';C:/Users/Rade/Anaconda3/Library/bin'

import numpy as np
import pandas as pd
import util
import jrboost

dataPath = r'C:/Users/Rade/Documents/Data Analysis/Data/Iris/Iris.csv'
dataFrame = pd.read_csv(dataPath, sep = ',', index_col = 0)
outDataFrame = util.oneHotEncode(dataFrame['Species'])
inDataFrame = dataFrame.drop(['Species'], axis = 1)

print(inDataFrame.head(5))
print()
print(outDataFrame.head(5))
print()

sampleCount = len(dataFrame.index)

#label = 'Iris-setosa'
label = 'Iris-versicolor'

inData = inDataFrame.to_numpy()
outData = outDataFrame[label].to_numpy();
weights = np.full((sampleCount,), 1.0)

opt = jrboost.StubOptions()
opt.usedSampleRatio = 1.0
opt.usedVariableRatio = 1.0

trainer = opt.createTrainer()
trainer.setInData(inData)
trainer.setOutData(outData)
trainer.setWeights(weights)

predictor = trainer.train()
predOutData = predictor.predict(inData)

print()
print(list(zip(outData, predOutData)))



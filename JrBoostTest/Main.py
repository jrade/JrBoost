import os
os.environ['PATH'] += ';C:/Users/Rade/Anaconda3/Library/bin'

import numpy as np
import jrboost

sampleCount = 5
variableCount = 12

trainInData = np.random.rand(sampleCount, variableCount)
trainOutData = np.random.rand(sampleCount)
trainWeights = np.random.rand(sampleCount)
testInData = np.random.rand(sampleCount, variableCount);

opt = jrboost.StubOptions()

trainer = opt.createTrainer()
trainer.setInData(trainInData)
trainer.setOutData(trainOutData)
trainer.setWeights(trainWeights)

predictor = trainer.train()
assert predictor.variableCount() == variableCount

predOutData = predictor.predict(testInData)
assert predOutData.shape == (sampleCount,)

print('Done!')

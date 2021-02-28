import copy, sys
import numpy as np

sys.path += ['../JrBoost/JrBoostTest', '../JrBoost/x64/Release']
import util
import jrboost


def optimize(cvParam, evalFun):

    bestOptionCount = cvParam['bestOptionCount']
    ultraBoost = cvParam.get('ultraBoost', None)
    bagSize = cvParam['bagSize']

    bpValues = cvParam['boostParamValues']

    optionList = [jrboost.BoostOptions()]
    for name, values in bpValues.items():
        tmp = []
        for opt in optionList:
            for value in values:
                setattr(opt, name, value)
                tmp.append(copy.copy(opt))
        optionList = tmp

    loss = evalFun(optionList)
    
    bestIndices = list(np.argsort(loss)) 
    del bestIndices[bestOptionCount:]
    bestOptionList = [optionList[i] for i in bestIndices]

    if ultraBoost is not None and ultraBoost != 1:
        for opt in bestOptionList:
            opt.iterationCount *= ultraBoost
            opt.eta /= ultraBoost

    return bagSize * bestOptionList

import copy, itertools, random
import numpy as np
import jrboost


def optimize(cvParam, evalFun):

    populationCount = cvParam['populationCount']
    survivorCount = cvParam['survivorCount']
    cycleCount = cvParam['cycleCount']

    bestOptionCount = cvParam['bestOptionCount']
    ultraBoost = cvParam.get('ultraBoost', None)
    bagSize = cvParam['bagSize']

    bpValues = copy.deepcopy(cvParam['boostParamValues'])

    for k in itertools.count():

        for values in bpValues.values():
            valueCount = len(values)
            values *= (populationCount + valueCount - 1) // valueCount
            random.shuffle(values)
            del values[populationCount:]

        optionList = [jrboost.BoostOptions() for _ in range(populationCount)]
        for name, values in bpValues.items():
            for i in range(populationCount):
                setattr(optionList[i], name, values[i])

        loss = evalFun(optionList)
        sortedIndices = list(np.argsort(loss))

        k += 1
        if k == cycleCount:
            del sortedIndices[bestOptionCount:]
            optionList = [optionList[i] for i in sortedIndices]
            if ultraBoost is not None and ultraBoost != 1:
                for opt in optionList:
                    opt.iterationCount *= ultraBoost
                    opt.eta /= ultraBoost
            return bagSize * optionList

        del sortedIndices[survivorCount:]
        for values in bpValues.values():
            values[:] = [values[i] for i in sortedIndices]
            values.sort()

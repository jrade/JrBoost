#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import copy, itertools, random
import numpy as np
import jrboost


def optimize(cvParam, evalFun):

    cycleCount = cvParam['cycleCount']
    populationCount = cvParam['populationCount']
    survivorCount = cvParam['survivorCount']
    bpValues = cvParam['boostParamValues']

    bpValues = copy.deepcopy(bpValues)
    k = 0
    while True:

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
            break

        del sortedIndices[survivorCount:]
        for values in bpValues.values():
            values[:] = [values[i] for i in sortedIndices]
            values.sort()

    # finalize

    bestOptionCount = cvParam['bestOptionCount']
    ultraBoost = cvParam.get('ultraBoost', None)
    bagSize = cvParam.get('bagSize', None)

    del sortedIndices[bestOptionCount:]
    optionList = [optionList[i] for i in sortedIndices]

    if ultraBoost is not None:
        for opt in optionList:
            opt.iterationCount *= ultraBoost
            opt.eta /= ultraBoost

    if bagSize is not None:
        optionList *= bagSize

    return optionList

#  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

import copy
import numpy as np
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

    if ultraBoost is not None:
        for opt in bestOptionList:
            opt.iterationCount *= ultraBoost
            opt.eta /= ultraBoost

    if bagSize is not None:
        optionList *= bagSize

    return optionList

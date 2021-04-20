//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BasePredictor.h"


class TrivialPredictor : public BasePredictor {
public:
    TrivialPredictor(size_t variableCount, double y) : 
        BasePredictor(variableCount),
        y_(y)
    {
        ASSERT(std::isfinite(y));
    }

    virtual ~TrivialPredictor() = default;

private:
    virtual void predictImpl_(CRefXXf inData, double c, RefXd outData) const
    {
        outData += c * y_;
    }

    double y_;
};

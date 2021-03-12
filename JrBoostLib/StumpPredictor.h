//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "SimplePredictor.h"


class StumpPredictor : public SimplePredictor {
public:
    virtual ~StumpPredictor() = default;

    virtual void predict(CRefXXf inData, double c, RefXd outData) const;

private:
    template<typename SampleIndex> friend class StumpTrainerImpl;

    StumpPredictor(size_t variableCount, size_t j, float x, double leftY, double rightY);

    size_t j_;
    float x_;
    double leftY_;
    double rightY_;
};

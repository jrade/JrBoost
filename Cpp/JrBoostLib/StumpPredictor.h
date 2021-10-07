//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BasePredictor.h"


class StumpPredictor : public BasePredictor
{
public:
    StumpPredictor(size_t j, float x, float leftY, float rightY, float gain = 0.0f);
    virtual ~StumpPredictor() = default;

private:
    virtual void predict_(CRefXXfc inData, double c, RefXd outData) const;
    virtual void variableWeights_(double c, RefXd weights) const;
    virtual void save_(ostream& os) const;

    friend class BasePredictor;
    static unique_ptr<BasePredictor> load_(istream& is, int version);

private:
    size_t j_;
    float x_;
    float leftY_;
    float rightY_;
    float gain_;
};

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BasePredictor.h"


class StumpPredictor : public BasePredictor {
public:
    StumpPredictor(size_t j, float x, double leftY, double rightY);
    virtual ~StumpPredictor() = default;

private:
    virtual void predict_(CRefXXf inData, double c, RefXd outData) const;

    virtual void save_(ostream& os) const;

    friend class BasePredictor;
    static unique_ptr<BasePredictor> load_(istream& is, int version);

private:
    uint32_t j_;
    float x_;
    float leftY_;
    float rightY_;
};

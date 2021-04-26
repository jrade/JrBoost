//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BasePredictor.h"


class TrivialPredictor : public BasePredictor {
public:
    TrivialPredictor(double y);
    virtual ~TrivialPredictor() = default;

private:
    virtual void predict_(CRefXXf inData, double c, RefXd outData) const;
    
    friend class BasePredictor;
    static unique_ptr<BasePredictor> load_(istream& is);

    virtual void save_(ostream& os) const;

    float y_;
};

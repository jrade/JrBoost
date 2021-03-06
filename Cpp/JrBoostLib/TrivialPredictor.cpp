//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TrivialPredictor.h"


TrivialPredictor::TrivialPredictor(double y) :
    y_(static_cast<float>(y))
{
    ASSERT(std::isfinite(y));
}


void TrivialPredictor::predict_(CRefXXf inData, double c, RefXd outData) const
{
    (void)inData;
    outData += c * static_cast<double>(y_);
}


void TrivialPredictor::save_(ostream& os) const
{
    const int type = Trivial;
    os.put(static_cast<char>(type));

    os.write(reinterpret_cast<const char*>(&y_), sizeof(y_));
}

unique_ptr<BasePredictor> TrivialPredictor::load_(istream& is, int version)
{
    if (version < 2) is.get();

    float y;
    is.read(reinterpret_cast<char*>(&y), sizeof(y));
    return std::make_unique<TrivialPredictor>(y);
}

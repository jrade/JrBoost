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
    outData += c * static_cast<double>(y_);
}


void TrivialPredictor::save_(ostream& os) const
{
    const uint8_t type = Trivial;
    os.put(static_cast<char>(type));

    const uint8_t version = 1;
    os.put(static_cast<char>(version));

    os.write(reinterpret_cast<const char*>(&y_), sizeof(y_));
}

unique_ptr<BasePredictor> TrivialPredictor::load_(istream& is)
{
    uint8_t version = static_cast<uint8_t>(is.get());
    if (version != 1)
        throw runtime_error("Not a valid JrBoost predictor file.");

    float y;
    is.read(reinterpret_cast<char*>(&y), sizeof(y));
    return std::make_unique<TrivialPredictor>(y);
}

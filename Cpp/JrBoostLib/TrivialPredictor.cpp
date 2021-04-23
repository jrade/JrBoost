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


void TrivialPredictor::predictImpl_(CRefXXf inData, double c, RefXd outData) const
{
    outData += c * static_cast<double>(y_);
}


void TrivialPredictor::save(ostream& os) const
{
    const uint8_t type = Trivial;
    os.write(reinterpret_cast<const char*>(&type), sizeof(type));

    const uint8_t version = 1;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));

    os.write(reinterpret_cast<const char*>(&y_), sizeof(y_));
}

unique_ptr<BasePredictor> TrivialPredictor::loadImpl_(istream& is)
{
    uint8_t version;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1)
        throw runtime_error("Not a valid JrBoost predictor file.");

    float y;
    is.read(reinterpret_cast<char*>(&y), sizeof(y));
    return std::make_unique<TrivialPredictor>(y);
}

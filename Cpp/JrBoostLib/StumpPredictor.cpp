//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "StumpPredictor.h"


StumpPredictor::StumpPredictor(size_t j, float x, double leftY, double rightY) :
    j_{ static_cast<uint32_t>(j) },
    x_{ x },
    leftY_{ static_cast<float>(leftY) },
    rightY_{ static_cast<float>(rightY) }
{
    ASSERT(std::isfinite(x) && std::isfinite(leftY) && std::isfinite(rightY));
}


void StumpPredictor::predict_(CRefXXf inData, double c, RefXd outData) const
{
    const size_t sampleCount = inData.rows();
    for (size_t i = 0; i < sampleCount; ++i) {
        double y = (inData(i, j_) < x_) ? leftY_ : rightY_;
        outData(i) += c * y;
    }
}


void StumpPredictor::save(ostream& os) const
{
    const uint8_t type = Stump;
    os.put(static_cast<char>(type));

    const uint8_t version = 1;
    os.put(static_cast<char>(version));

    os.write(reinterpret_cast<const char*>(&j_), sizeof(j_));
    os.write(reinterpret_cast<const char*>(&x_), sizeof(x_));
    os.write(reinterpret_cast<const char*>(&leftY_), sizeof(leftY_));
    os.write(reinterpret_cast<const char*>(&rightY_), sizeof(rightY_));
}

unique_ptr<BasePredictor> StumpPredictor::load_(istream& is)
{
    uint8_t version = static_cast<uint8_t>(is.get());
    if (version != 1)
        throw runtime_error("Not a valid JrBoost predictor file.");

    uint32_t j;
    float x;
    float leftY;
    float rightY;
    is.read(reinterpret_cast<char*>(&j), sizeof(j));
    is.read(reinterpret_cast<char*>(&x), sizeof(x));
    is.read(reinterpret_cast<char*>(&leftY), sizeof(leftY));
    is.read(reinterpret_cast<char*>(&rightY), sizeof(rightY));
    return unique_ptr<StumpPredictor>(new StumpPredictor(j, x, leftY, rightY));
}

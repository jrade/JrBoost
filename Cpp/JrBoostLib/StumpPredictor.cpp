//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "StumpPredictor.h"


StumpPredictor::StumpPredictor(uint32_t j, float x, float leftY, float rightY, float gain) :
    j_{ j },
    x_{ x },
    leftY_{ leftY },
    rightY_{ rightY },
    gain_{ gain }
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


void StumpPredictor::save_(ostream& os) const
{
    const int type = Stump;
    os.put(static_cast<char>(type));

    os.write(reinterpret_cast<const char*>(&j_), sizeof(j_));
    os.write(reinterpret_cast<const char*>(&x_), sizeof(x_));
    os.write(reinterpret_cast<const char*>(&leftY_), sizeof(leftY_));
    os.write(reinterpret_cast<const char*>(&rightY_), sizeof(rightY_));
    os.write(reinterpret_cast<const char*>(&gain_), sizeof(gain_));
}

unique_ptr<BasePredictor> StumpPredictor::load_(istream& is, int version)
{
    if (version < 2) is.get();

    uint32_t j;
    float x;
    float leftY;
    float rightY;
    float gain;
    is.read(reinterpret_cast<char*>(&j), sizeof(j));
    is.read(reinterpret_cast<char*>(&x), sizeof(x));
    is.read(reinterpret_cast<char*>(&leftY), sizeof(leftY));
    is.read(reinterpret_cast<char*>(&rightY), sizeof(rightY));
    if (version >= 3)
        is.read(reinterpret_cast<char*>(&gain), sizeof(gain));

    return std::make_unique<StumpPredictor>(j, x, leftY, rightY, gain);
}

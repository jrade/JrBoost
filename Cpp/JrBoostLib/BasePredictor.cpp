//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BasePredictor.h"
#include "TrivialPredictor.h"
#include "StumpPredictor.h"


void BasePredictor::predict(CRefXXf inData, double c, RefXd outData) const
{
    PROFILE::PUSH(PROFILE::BOOST_PREDICT);
    predictImpl_(inData, c, outData);
    size_t sampleCount = inData.rows();
    PROFILE::POP(sampleCount);
}


unique_ptr<BasePredictor> BasePredictor::load(istream& is)
{
    uint8_t type = static_cast<uint8_t>(is.get());
    if (type == Trivial)
        return TrivialPredictor::loadImpl_(is);
    if (type == Stump)
        return StumpPredictor::loadImpl_(is);
    throw runtime_error("Not a valid JrBoost predictor file.");
}

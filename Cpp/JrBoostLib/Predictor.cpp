//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "Predictor.h"
#include "BoostPredictor.h"
#include "EnsemblePredictor.h"


ArrayXd Predictor::predict(CRefXXf inData) const
{
    PROFILE::PUSH(PROFILE::BOOST_PREDICT);
    ArrayXd pred = predictImpl_(inData);
    size_t sampleCount = inData.rows();
    PROFILE::POP(sampleCount);
    return pred;
}


void Predictor::validateInData(CRefXXf inData) const
{
    ASSERT(static_cast<size_t>(inData.cols()) == variableCount());
    //ASSERT((inData > -numeric_limits<float>::infinity()).all());
    //ASSERT((inData < numeric_limits<float>::infinity()).all());
}

shared_ptr<Predictor> Predictor::load(const string& filePath)
{
    ifstream ifs(filePath, std::ios::binary);
    if (!ifs)
        throw runtime_error("Unable to open the file " + filePath + " for reading.");
        return load(ifs);
}

shared_ptr<Predictor> Predictor::load(istream& is)
{    
    uint8_t type = static_cast<uint8_t>(is.get());
    if (type == Boost)
        return BoostPredictor::loadImpl_(is);
    if (type == Ensemble)
        return EnsemblePredictor::loadImpl_(is);
    throw runtime_error("Not a valid JrBoost predictor file.")
}

void Predictor::save(const string& filePath) const
{
    ofstream ofs(filePath, std::ios::binary);
    if (!ofs)
        throw runtime_error("Unable to open the file " + filePath + " for writing.");
    return save(ofs);
}

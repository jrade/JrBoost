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

    validateInData_(inData);
    ArrayXd pred = predict_(inData);

    size_t sampleCount = inData.rows();
    PROFILE::POP(sampleCount);

    return pred;
}


void Predictor::validateInData_(CRefXXf inData) const
{
    PROFILE::PUSH(PROFILE::VALIDATE);
    size_t ITEM_COUNT = inData.rows() * inData.cols();

    ASSERT(static_cast<size_t>(inData.cols()) == variableCount());
    ASSERT((inData.abs() < numeric_limits<float>::infinity()).all());

    PROFILE::POP(ITEM_COUNT);
}


void Predictor::save(const string& filePath) const
{
    ofstream ofs(filePath, std::ios::binary);
    if (!ofs)
        throw runtime_error("Unable to open the file " + filePath + " for writing.");

    ofs.write("JRBOOST", 7);

    const uint8_t version = 1;
    ofs.put(static_cast<char>(version));

    save_(ofs);
}

shared_ptr<Predictor> Predictor::load(const string& filePath)
{
    ifstream ifs(filePath, std::ios::binary);
    if (!ifs)
        throw runtime_error("Unable to open the file " + filePath + " for reading.");

    char sig[7];
    ifs.read(sig, 7);
    if (memcmp(sig, "JRBOOST", 7) != 0)
        throw runtime_error("Not a valid JrBoost predictor file.");

    uint8_t version = static_cast<uint8_t>(ifs.get());
    if (version != 1)
        throw runtime_error("Not a valid JrBoost predictor file.");

    return load_(ifs);
}

shared_ptr<Predictor> Predictor::load_(istream& is)
{
    uint8_t type = static_cast<uint8_t>(is.get());
    switch (type) {
    case Boost:
        return BoostPredictor::load_(is);
    case Ensemble:
        return EnsemblePredictor::load_(is);
    default:
        throw runtime_error("Not a valid JrBoost predictor file.");
    }
}


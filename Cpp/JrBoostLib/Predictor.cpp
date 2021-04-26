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

    ASSERT(static_cast<size_t>(inData.cols()) == variableCount());
    ASSERT((inData.abs() < numeric_limits<float>::infinity()).all());

    ArrayXd pred = predict_(inData);

    size_t sampleCount = inData.rows();
    PROFILE::POP(sampleCount);

    return pred;
}


void Predictor::save(const string& filePath) const
{
    ofstream ofs;
    ofs.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    ofs.open(filePath, std::ios::binary);
    save(ofs);
}

void Predictor::save(ostream& os) const
{
    os.write("JRBOOST", 7);
    const uint8_t version = 1;
    os.put(static_cast<char>(version));
    save_(os);
    os.put('!');
}

shared_ptr<Predictor> Predictor::load(const string& filePath)
{
    ifstream ifs;
    ifs.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    ifs.open(filePath, std::ios::binary);
    try {
        return load(ifs);
    }
    catch(std::ios::failure&) {
        throw runtime_error("Not a valid JrBoost predictor file.");
    }
}

shared_ptr<Predictor> Predictor::load(istream& is)
{
    char sig[7];
    is.read(sig, 7);
    if (memcmp(sig, "JRBOOST", 7) != 0)
        throw runtime_error("Not a JrBoost predictor file.");

    uint8_t version = static_cast<uint8_t>(is.get());
    if (version < 1)
        throw runtime_error("Not a valid JrBoost predictor file.");
    if (version > 1)
        throw runtime_error("Reading this JrBoost predictor file requires a newer version of the JrBoost library.");
    
    shared_ptr<Predictor> pred = load_(is);

    if (is.get() != '!')
        throw runtime_error("Not a valid JrBoost predictor file.");

    return pred;
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

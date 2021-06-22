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

    if (static_cast<size_t>(inData.cols()) != variableCount())
        throw std::invalid_argument("Train and test indata have different numbers of variables.");
    if (!(inData.abs() < numeric_limits<float>::infinity()).all())
        throw std::invalid_argument("Test indata has values that are infinity or NaN.");

    ArrayXd pred = predict_(inData);

    size_t sampleCount = inData.rows();
    PROFILE::POP(sampleCount);

    return pred;
}

//----------------------------------------------------------------------------------------------------------------------

// File format versions:
// 1 - original version
// 2 - added tree predictors, simplified version handling

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
    os.put(static_cast<char>(currentVersion_));
    save_(os);
    os.put('!');
}

shared_ptr<Predictor> Predictor::load(const string& filePath)
{
    ifstream ifs;
    ifs.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    ifs.open(filePath, std::ios::binary);
    return load(ifs);
}

shared_ptr<Predictor> Predictor::load(istream& is)
{
    char sig[7];
    is.read(sig, 7);
    if (memcmp(sig, "JRBOOST", 7) != 0)
        throw std::runtime_error("Not a JrBoost predictor file.");

    int version = is.get();
    if (version < 1)
        parseError_(is);
    if (version > currentVersion_)
        throw std::runtime_error("Reading this JrBoost predictor file requires a newer version of the JrBoost library.");

    shared_ptr<Predictor> pred = load_(is, version);

    if (is.get() != '!')
        parseError_(is);

    return pred;
}

shared_ptr<Predictor> Predictor::load_(istream& is, int version)
{
    int type = is.get();
    if (type == Boost)
        return BoostPredictor::load_(is, version);
    if (type == Ensemble)
        return EnsemblePredictor::load_(is, version);
    parseError_(is);
}

void Predictor::parseError_ [[noreturn]]  (istream& is)
{
    string msg = "Not a valid JrBoost predictor file.";

    int64_t pos = is.tellg();
    if (pos != -1)
        msg += "\n(Parsing error after " + std::to_string(pos) + " bytes)";

    throw std::runtime_error(msg);
}

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BasePredictor.h"

#include "TreePredictor.h"
#include "TrivialPredictor.h"
#include "StumpPredictor.h"


void BasePredictor::predict(CRefXXfc inData, double c, RefXd outData) const
{
    PROFILE::PUSH(PROFILE::TREE_PREDICT);
    predict_(inData, c, outData);
    PROFILE::POP();
}

unique_ptr<BasePredictor> BasePredictor::load_(istream& is, int version)
{
    int type = is.get();
    if (type == Trivial)
        return TrivialPredictor::load_(is, version);
    if (type == Stump)
        return StumpPredictor::load_(is, version);
    if (version >= 2 && type == Tree)
        return TreePredictor::load_(is,version);
    parseError_(is);
}

void BasePredictor::parseError_ [[noreturn]] (istream& is)
{
    string msg = "Not a valid JrBoost predictor file.";

    int64_t pos = is.tellg();
    if (pos != -1)
        msg += "\n(Parsing error after " + std::to_string(pos) + " bytes)";

    throw std::runtime_error(msg);
}

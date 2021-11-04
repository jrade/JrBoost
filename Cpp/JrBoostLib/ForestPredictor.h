//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BasePredictor.h"


class ForestPredictor : public BasePredictor
{
public:
    ForestPredictor(vector<unique_ptr<BasePredictor>>&& basePredictors) :
        basePredictors_(move(basePredictors))
    {}

    virtual ~ForestPredictor() = default;

private:
    virtual void predict_(CRefXXfc inData, double c, RefXd outData) const
    {
        c /= size(basePredictors_);
        for (const auto& basePredictor : basePredictors_)
            basePredictor->predict_(inData, c, outData);
    }

    virtual void variableWeights_(double c, RefXd weights) const
    {
        c /= size(basePredictors_);
        for (const auto& basePredictor : basePredictors_)
            basePredictor->variableWeights_(c, weights);
    }

    void save_(ostream& os) const
    {
        const int type = Forest;
        os.put(static_cast<char>(type));

        const uint32_t n = static_cast<uint32_t>(size(basePredictors_));
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (uint32_t i = 0; i < n; ++i)
            basePredictors_[i]->save_(os);
    }

    friend class BasePredictor;
    static unique_ptr<BasePredictor> load_(istream& is, int version)
    {
        uint32_t n;
        is.read(reinterpret_cast<char*>(&n), sizeof(n));
        vector<unique_ptr<BasePredictor>> basePredictors(n);
        for (uint32_t i = 0; i < n; ++i)
            basePredictors[i] = BasePredictor::load_(is, version);

        return std::make_unique<ForestPredictor>(move(basePredictors));
    }

private:
    const vector<unique_ptr<BasePredictor>> basePredictors_;
};

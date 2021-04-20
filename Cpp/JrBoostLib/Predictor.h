//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class Predictor {
public:
    virtual ~Predictor() = default;

    virtual size_t variableCount() const = 0;

    ArrayXd predict(CRefXXf inData) const
    {
        PROFILE::PUSH(PROFILE::BOOST_PREDICT);
        ArrayXd pred = predictImpl_(inData);
        size_t sampleCount = inData.rows();
        PROFILE::POP(sampleCount);
        return pred;
    }

protected:
    Predictor() = default;

    void validateInData(CRefXXf inData) const
    {
        ASSERT(static_cast<size_t>(inData.cols()) == variableCount());
        //ASSERT((inData > -numeric_limits<float>::infinity()).all());
        //ASSERT((inData < numeric_limits<float>::infinity()).all());
    }

private:
    friend class EnsemblePredictor;

    virtual ArrayXd predictImpl_(CRefXXf inData) const = 0;

// deleted:
    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor&) = delete;
};

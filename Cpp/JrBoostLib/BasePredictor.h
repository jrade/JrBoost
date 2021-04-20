//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class BasePredictor {
public:
    virtual ~BasePredictor() = default;

    size_t variableCount() const { return variableCount_; }

    void predict(CRefXXf inData, double c, RefXd outData) const
    {
        PROFILE::PUSH(PROFILE::BOOST_PREDICT);
        predictImpl_(inData, c, outData);
        size_t sampleCount = inData.rows();
        PROFILE::POP(sampleCount);
    }

protected:
    BasePredictor(size_t variableCount) : variableCount_(variableCount) {}

private:
    virtual void predictImpl_(CRefXXf inData, double c, RefXd outData) const = 0;

    size_t variableCount_;
    
// deleted:
    BasePredictor(const BasePredictor&) = delete;
    BasePredictor& operator=(const BasePredictor&) = delete;
};

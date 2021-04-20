//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class BasePredictor {
public:
    virtual ~BasePredictor() = default;
    size_t variableCount() const { return variableCount_; }
    virtual void predict(CRefXXf inData, double c, RefXd outData) const = 0;

protected:
    BasePredictor(size_t variableCount) : variableCount_(variableCount) {}

// deleted:
    BasePredictor(const BasePredictor&) = delete;
    BasePredictor& operator=(const BasePredictor&) = delete;

private:
    size_t variableCount_;
};

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class SimplePredictor {
public:
    virtual ~SimplePredictor() = default;
    size_t variableCount() const { return variableCount_; }
    virtual void predict(CRefXXf inData, double c, RefXd outData) const = 0;

protected:
    SimplePredictor(size_t variableCount) : variableCount_(variableCount) {}

// deleted:
    SimplePredictor(const SimplePredictor&) = delete;
    SimplePredictor& operator=(const SimplePredictor&) = delete;

private:
    size_t variableCount_;
};

//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Tools.h"


template<typename Value, size_t N>
class StaticStack {
public:
    StaticStack() = default;
    ~StaticStack() = default;
    StaticStack(const StaticStack&) = default;
    StaticStack& operator=(const StaticStack&) = default;

    bool empty() const { return pos_ == 0; }
    size_t size() const { return pos_; }

    Value& top()
    {
        ASSERT(pos_ > 0);
        return values_[pos_ - 1];
    }
    const Value& top() const
    {
        ASSERT(pos_ > 0);
        return values_[pos_ - 1];
    }

    void push(const Value value)
    {
        ASSERT(pos_ < N);
        values_[pos_] = value;
        ++pos_;
    }

    void pop()
    {
        ASSERT(pos_ > 0);
        --pos_;
    }

private:
    array<Value, N> values_;
    size_t pos_ = 0;
};

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "TreeOptions.h"


class ForestOptions : public TreeOptions
{
public:
    ForestOptions() = default;
    ForestOptions(const ForestOptions&) = default;
    ForestOptions& operator=(const ForestOptions&) = default;
    ~ForestOptions() = default;

    size_t forestSize() const { return forestSize_; }

    void setForestSize(size_t n)
    {
        if (n == 0)
            throw std::invalid_argument("forestSize must be positive.");
        forestSize_ = n;
    }

protected:
    double cost() const { return TreeOptions::cost() * forestSize_; }

private:
    size_t forestSize_{ 1 };
};

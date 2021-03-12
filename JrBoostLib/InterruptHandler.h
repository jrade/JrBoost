//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class InterruptHandler {
public:
    virtual void check() = 0;
};


inline InterruptHandler* currentInterruptHandler = nullptr;

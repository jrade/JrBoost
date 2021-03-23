//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Clock.h"
#include "StaticStack.h"


class PROFILE {
public:
    inline static bool doProfile = false;

    enum CLOCK_ID {
        MAIN, T_RANK, BOOST_TRAIN,
        STUMP_TRAIN, VALIDATE, USED_SAMPLES, USED_VARIABLES, SUMS, SORTED_USED_SAMPLES, BEST_SPLIT,
        BOOST_PREDICT, OMP_BARRIER, MEMORY,
        ZERO, CLOCK_COUNT
    };
    
    static void PUSH(CLOCK_ID id);
    static void POP(size_t itemCount = 0);
    static void SWITCH(size_t itemCount, CLOCK_ID id);
    static void PRINT();

    static uint64_t SPLIT_ITERATION_COUNT;
    static uint64_t SLOW_BRANCH_COUNT;

private:
    static Clock clocks_[CLOCK_COUNT];
    static const string names_[CLOCK_COUNT];
    static StaticStack<CLOCK_ID, 1000> clockIndexStack_;
    // Using a std::vector for the clockIndexStack_ would lead to problems:
    // (1) Static initialization problems: PROFILE::PUSH() may be called before clockIndexStack_ has been initialized
    // (2) Reentrancy if we profile operator new and delete:
    //       PROFILE::PUSH() -> vector::push() -> operator new() -> PROFILE::PUSH()
};


inline const string PROFILE::names_[PROFILE::CLOCK_COUNT] = {
    "main",
    "  t-rank",
    "  train boost",
    "  train stumps",
    "    validate",
    "    used samples",
    "    used variables",
    "    sums",
    "    sorted used s.",
    "    best split",
    "  predict",
    "  OMP barrier",
    "  dyn. memory",
    "zero"
};

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Clock.h"
#include "StaticStack.h"


class PROFILE {

public:
    static void START()
    {
        ENABLED = true;
        PUSH(MAIN);
    }

    static string STOP()
    {
        POP(MAIN);
        string msg = RESULT();
        ENABLED = false;
        return msg;
    }

public:
    inline static bool ENABLED = false;

    enum CLOCK_ID {
        MAIN, 
        T_RANK, BOOST_INIT, STUMP_INIT, BOOST_TRAIN, STUMP_TRAIN,
        /*VALIDATE,*/ USED_SAMPLES, USED_VARIABLES, SUMS, SORTED_USED_SAMPLES, BEST_SPLIT, STUMP_PREDICT,
        THREAD_SYNCH, BOOST_PREDICT,
        TEST1, TEST2, TEST3,
        ZERO, CLOCK_COUNT
    };
    
    static void PUSH(CLOCK_ID id);
    static void POP(size_t itemCount = 0);
    static void SWITCH(size_t itemCount, CLOCK_ID id);
    static string RESULT();

private:
    static Clock clocks_[CLOCK_COUNT];
    static const string names_[CLOCK_COUNT];
    static StaticStack<CLOCK_ID, 1000> clockIndexStack_;

    template<typename SampleIndex> friend class StumpTrainerImpl;
    static uint64_t SPLIT_ITERATION_COUNT;
    static uint64_t SLOW_BRANCH_COUNT;
};


inline const string PROFILE::names_[PROFILE::CLOCK_COUNT] = {
    "main",
    "  t-rank",
    "  boost init",
    "    stump init",
    "  boost train",
    "    stump train",
//  "      validate",
    "      used samples",
    "      used variables",
    "      sums",
    "      sorted used s.",
    "      best split",
    "    stump predict",
    "    thread synch",
    "  boost predict",
    "  test-1",
    "  test-2",
    "  test-3",
    "zero"
};

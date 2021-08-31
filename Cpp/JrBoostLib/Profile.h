//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Clock.h"
#include "StaticStack.h"


class PROFILE {
public:
    enum CLOCK_ID {
        MAIN, 
            T_RANK,
            BOOST_INIT,
            BOOST_TRAIN, 
                TREE_TRAIN,
                    /*VALIDATE,*/
                    INIT_USED_VARIABLES,
                    INIT_SAMPLE_STATUS,
                    UPDATE_SAMPLE_STATUS,
                    INIT_ORDERED_SAMPLES,
                    UPDATE_ORDERED_SAMPLES,
                    UPDATE_SPLITS,
                        SUMS,
                    UPDATE_TREE,
                    CREATE_PREDICTOR,
                TREE_PREDICT,
                THREAD_SYNCH, 
            BOOST_PREDICT,
        TEST1, TEST2, TEST3, TEST4, TEST5,
        ZERO, CLOCK_COUNT
    };
    
    static void START();
    static string STOP();
    static void PUSH(CLOCK_ID id);
    static void POP(size_t itemCount = 0);
    static void SWITCH(size_t itemCount, CLOCK_ID id);
    static void UPDATE_BRANCH_STATISTICS(size_t iterationCount, size_t slowBranchCount);

private:
    static string result_();

    static bool enabled_;
    static Clock clocks_[CLOCK_COUNT];
    static const string names_[CLOCK_COUNT];
    static StaticStack<CLOCK_ID, 1000> clockIndexStack_;
    static uint64_t splitIterationCount_;
    static uint64_t slowBranchCount_;
};


inline const string PROFILE::names_[PROFILE::CLOCK_COUNT] = {
    "main",
    "  t-rank",
    "  boost init",
    "  boost train",
    "    tree train",
//  "      validate",
    "      init used variables",
    "      init sample status",
    "      update sample status",
    "      init ordered samples",
    "      update ordered samples",
    "      update splits",
    "        sums",
    "      update tree",
    "      create predictor",
    "    tree predict",
    "    thread synch",
    "  boost predict",
    "  test-1",
    "  test-2",
    "  test-3",
    "  test-4",
    "  test-5",
    "zero"
};

//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
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
        F_RANK,
        BOOST_TRAIN,
        TREE_TRAIN,
        PACK_DATA,
        USED_VARIABLES,
        INIT_TREE,
        UPDATE_TREE,
        INIT_SAMPLE_STATUS,
        UPDATE_SAMPLE_STATUS,
        INIT_ORDERED_SAMPLES,
        UPDATE_ORDERED_SAMPLES,
        FIND_BEST_SPLITS,
        PREDICT,
        LOSS,
        INNER_THREAD_SYNCH,
        OUTER_THREAD_SYNCH,
        TEST1,
        TEST2,
        TEST3,
        TEST4,
        TEST5,
        ZERO,
        CLOCK_COUNT
    };

    static void START();
    static string STOP();
    static void PUSH(CLOCK_ID id);
    static void POP(size_t itemCount = 0);
    static void SWITCH(CLOCK_ID id, size_t itemCount = 0);
    static void UPDATE_BRANCH_STATISTICS(size_t iterationCount, size_t slowBranchCount);

private:
    static string result_();
    static string formatByteCount_(size_t n);

    static bool enabled_;
    static Clock clocks_[CLOCK_COUNT];
    static const string names_[CLOCK_COUNT];
    static StaticStack<CLOCK_ID, 1000> clockIndexStack_;
    static uint64_t splitIterationCount_;
    static uint64_t slowBranchCount_;
};


inline const string PROFILE::names_[PROFILE::CLOCK_COUNT]
    = {"main",
       "t-rank",
       "F-rank",
       "  boost train",
       "  tree train",
       "  pack data",
       "  used variables",
       "  init tree",
       "  update tree",
       "  init sample status",
       "  update sample status",
       "    init ord. samples",
       "    update ord. samples",
       "    find best splits",
       "  predict",
       "  loss",
       "inner thread synch",
       "outer thread synch",
       "test-1",
       "test-2",
       "test-3",
       "test-4",
       "test-5",
       "zero"};

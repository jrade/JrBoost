#pragma once

#include "Clock.h"
#include "StaticStack.h"

#define DO_PROFILE

class PROFILE {
public:
    enum CLOCK_ID {
        MAIN, T_RANK, BOOST_TRAIN,
        VALIDATE, USED_SAMPLES, USED_VARIABLES, SUMS, SORTED_USED_SAMPLES, BEST_SPLIT,
        LCP_P, OMP_BARRIER, MEMORY,
        CLOCK_COUNT
    };
    
    static void PUSH(CLOCK_ID id);
    static void POP(size_t itemCount = 0);
    static void SWITCH(size_t itemCount, CLOCK_ID id);
    static void PRINT();

    static uint64_t SPLIT_ITERATION_COUNT;
    static uint64_t SLOW_BRANCH_COUNT;

private:
    static void push_(CLOCK_ID id);
    static void pop_(size_t itemCount);
    static void switch_(size_t itemCount, CLOCK_ID id);

    static Clock clocks_[CLOCK_COUNT + 1];
    static const string names_[CLOCK_COUNT];
    static StaticStack<CLOCK_ID, 1000> clockIndexStack_;
    // Using a std::vector for the clock index stack would lead to problems if we log new and delete.
    // (1) Static initialization problems: PROFILE::PUSH() may be called before the std::vector has been initialized
    // (2) Reentrant calls: PROFILE::PUSH() -> vector::push() -> operator new() -> PROFILE::PUSH()
    static size_t i_;
};

inline const string PROFILE::names_[PROFILE::CLOCK_COUNT] = {
    "main",
    "  t-rank",
    "  train boost",
    "    validate",
    "    used samples",
    "    used variables",
    "    sums",
    "    sorted used s.",
    "    best split",
    "  predict",
    "  OMP barrier",
    "  dyn. memory",
};

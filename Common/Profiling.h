#pragma once

#include "ClockCycleCount.h"


namespace CLOCK {

    enum ID { MAIN, T_RANK, BT_TRAIN, ST_TRAIN, ST_VAL, /*T1, T2,*/ LCP_P, OMP_BARRIER, MEMORY, CLOCK_COUNT };

    inline void PUSH(int i);
    inline void POP(size_t itemCount = 0);
    inline void PRINT();

    inline const string names[CLOCK_COUNT] = {
        "main",
        "  t-rank",
        "  train boost",
        "    train stump",
        "      validate",
        //"      T1",
       // "      T2",
        "  predict",
        "  OMP barrier",
        "  dyn. memory",
    };

    inline void log(string name, size_t clockCount, size_t totalClockCount, size_t itemCount)
    {
        cout << std::setw(18) << std::left << name << std::right;
        cout << std::setw(8) << std::fixed << std::setprecision(0) << 1e-6 * clockCount;
        cout << "  ";
        cout << std::setw(4) << std::setprecision(1) << (100.0 * clockCount) / totalClockCount << "%";
        if (itemCount != 0) {
            cout << "  ";
            cout << std::setw(5) << static_cast<double>(clockCount) / itemCount;
        }
        cout << endl;
    }


    // implementation --------------------------------------------------------------------------------------------------

    inline uint64_t clockCounts_[CLOCK_COUNT] = { 0 };
    inline uint64_t itemCounts_[CLOCK_COUNT] = { 0 };

    const int clockStackSize_ = 256;
    inline int clockIndexStack_[clockStackSize_];
    inline int stackIndex_ = -1;

    inline void PUSH(int i)
    {
#pragma omp master
        {
            uint64_t t = clockCycleCount();
            if (stackIndex_ != -1) {
                int prevI = clockIndexStack_[stackIndex_];
                clockCounts_[prevI] += t;
            }
            ++stackIndex_;
            ASSERT(stackIndex_ != clockStackSize_);
            clockIndexStack_[stackIndex_] = i;
            clockCounts_[i] -= t;
        }
    }

    inline void POP(size_t itemCount)
    {
#pragma omp master
        {
            uint64_t t = clockCycleCount();
            ASSERT(stackIndex_ != -1);
            int i = clockIndexStack_[stackIndex_];
            clockCounts_[i] += t;
            itemCounts_[i] += itemCount;
            --stackIndex_;
            if (stackIndex_ >= 0) {
                int prevI = clockIndexStack_[stackIndex_];
                clockCounts_[prevI] -= t;
            }
        }
    }

    inline void PRINT()
    {
        size_t totalClockCount = std::accumulate(std::begin(clockCounts_), std::end(clockCounts_), (uint64_t)0);

        for (int i = 0; i < CLOCK_COUNT; ++i)
            log(names[i], clockCounts_[i], totalClockCount, itemCounts_[i]);

        cout << endl;

        std::fill(std::begin(clockCounts_), std::end(clockCounts_), 0);
        std::fill(std::begin(itemCounts_), std::end(itemCounts_), 0);
    }
}

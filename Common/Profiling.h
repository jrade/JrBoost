#pragma once

#include "ClockCycleCount.h"

#define PROFILE


namespace CLOCK {

    enum ID {
        MAIN, T_RANK, BT_TRAIN,
        ST_TRAIN, ST_VAL, USED_SAMPLES, USED_VARIABLES, SUMS, SORTED_USED_SAMPLES, BEST_SPLIT,
        LCP_P, OMP_BARRIER, MEMORY, ZERO, CLOCK_COUNT
    };

    inline void PUSH(ID id);
    inline void POP(size_t itemCount = 0);
    inline void PRINT();

    inline const string names[CLOCK_COUNT] = {
        "main",
        "  t-rank",
        "  train boost",
        "    train stump",
        "      validate",
        "      used samples",
        "      used variables",
        "      sums",
        "      sorted used s.",
        "      best split",
        "  predict",
        "  OMP barrier",
        "  dyn. memory",
        "null",
    };

    inline void log(string name, int64_t clockCount, int64_t totalClockCount, size_t itemCount)
    {
        cout << std::setw(22) << std::left << name << std::right;
        cout << std::setw(8) << std::fixed << std::setprecision(0) << 1e-6 * clockCount;
        cout << "  ";
        cout << std::setw(4) << std::setprecision(1) << (100.0 * clockCount) / totalClockCount << "%";
        if (itemCount != 0) {
            cout << "  ";
            cout << std::setw(5) << static_cast<double>(clockCount) / itemCount;
        }
        cout << endl;
    }

    const int64_t fudge = 120;


    // implementation --------------------------------------------------------------------------------------------------

#ifdef PROFILE

    inline int64_t totalFudge = 0;

    inline uint64_t clockCounts_[CLOCK_COUNT] = { 0 };
    inline uint64_t itemCounts_[CLOCK_COUNT] = { 0 };

    const int clockStackSize_ = 256;
    inline ID clockIndexStack_[clockStackSize_];
    inline int stackIndex_ = -1;

    inline void PUSH(ID id)
    {
#pragma omp master
        {
            uint64_t t = clockCycleCount();
            if (stackIndex_ != -1) {
                ID prevId = clockIndexStack_[stackIndex_];
                clockCounts_[prevId] += t - fudge;
                totalFudge += fudge;
            }
            ++stackIndex_;
            ASSERT(stackIndex_ != clockStackSize_);
            clockIndexStack_[stackIndex_] = id;
            clockCounts_[id] -= t;
        }
    }

    inline void POP(size_t itemCount)
    {
#pragma omp master
        {
            uint64_t t = clockCycleCount();
            ASSERT(stackIndex_ != -1);
            ID id = clockIndexStack_[stackIndex_];
            clockCounts_[id] += t - fudge;
            totalFudge += fudge;
            itemCounts_[id] += itemCount;
            --stackIndex_;
            if (stackIndex_ >= 0) {
                ID prevId = clockIndexStack_[stackIndex_];
                clockCounts_[prevId] -= t;
            }
        }
    }

    inline void PRINT()
    {
        int64_t totalClockCount = std::accumulate(std::begin(clockCounts_), std::end(clockCounts_), (uint64_t)0);

        for (int i = 0; i < CLOCK_COUNT; ++i) {
            ID id = static_cast<ID>(i);
            log(names[id], clockCounts_[id], totalClockCount, itemCounts_[id]);
            clockCounts_[id] = 0;
            itemCounts_[id] = 0;
        }

        cout << "\nProfiling overhead: " << (100.0 * totalFudge) / (totalClockCount + totalFudge) << "%" << endl;
        totalFudge = 0;
    }

#else

    inline void PUSH(ID) {}
    inline void POP(size_t) {}
    inline void PRINT() {}

#endif

}


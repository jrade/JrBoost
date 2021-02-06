#pragma once

#include "ClockCycleCount.h"

#define PROFILE


namespace CLOCK {

    enum ID {
        MAIN, T_RANK, BT_TRAIN,
        ST_TRAIN, ST_VAL, USED_SAMPLES, USED_VARIABLES, SUMS, SORTED_USED_SAMPLES, BEST_SPLIT,
        LCP_P, OMP_BARRIER, MEMORY,
        CLOCK_COUNT
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
    };

    inline void log(string name, int64_t clockCount, uint64_t totalClockCount, size_t itemCount)
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

    inline uint64_t splitIterationCount = 0;
    inline uint64_t slowBranchCount = 0;


    // implementation --------------------------------------------------------------------------------------------------

#ifdef PROFILE

    inline uint64_t clockCounts_[CLOCK_COUNT + 1] = { 0 };
    inline uint64_t itemCounts_[CLOCK_COUNT + 1] = { 0 };
    inline uint64_t callCounts_[CLOCK_COUNT + 1] = { 0 };

    const int clockStackSize_ = 256;
    inline ID clockIndexStack_[clockStackSize_];
    inline int stackIndex_ = -1;

    inline void PUSH_IMPL(ID id)
    {
        uint64_t t = clockCycleCount();
        if (stackIndex_ != -1) {
            ID prevId = clockIndexStack_[stackIndex_];
            clockCounts_[prevId] += t;
            ++callCounts_[prevId];
        }
        ++stackIndex_;
        ASSERT(stackIndex_ != clockStackSize_);
        clockIndexStack_[stackIndex_] = id;
        clockCounts_[id] -= t;
    }

    inline void PUSH(ID id)
    {
        static size_t k = 0;
    #pragma omp master
        {
            if (++k % 100 == 0)
                PUSH_IMPL(CLOCK_COUNT);
            PUSH_IMPL(id);
        }
    }

    inline void POP_IMPL(size_t itemCount)
    {
        uint64_t t = clockCycleCount();
        ASSERT(stackIndex_ != -1);
        ID id = clockIndexStack_[stackIndex_];
        clockCounts_[id] += t;
        ++callCounts_[id];
        itemCounts_[id] += itemCount;
        --stackIndex_;
        if (stackIndex_ >= 0) {
            ID prevId = clockIndexStack_[stackIndex_];
            clockCounts_[prevId] -= t;
        }
    }

    inline void POP(size_t itemCount)
    {
#pragma omp master
        {
            POP_IMPL(itemCount);
            if (stackIndex_ >= 0 && clockIndexStack_[stackIndex_] == CLOCK_COUNT)
                POP_IMPL(0);
        }
    }


    inline void PRINT()
    {
        size_t zeroCalibration = clockCounts_[CLOCK_COUNT] / callCounts_[CLOCK_COUNT];
        for (int id = 0; id <= CLOCK_COUNT; ++id)
            clockCounts_[id] -= callCounts_[id] * zeroCalibration;

        const uint64_t totalClockCount = std::accumulate(std::begin(clockCounts_), std::end(clockCounts_), (uint64_t)0);
        const uint64_t totalCallCount = std::accumulate(std::begin(callCounts_), std::end(callCounts_), (uint64_t)0);
        const uint64_t totalZeroCalibration = totalCallCount * zeroCalibration;

        for (int id = 0; id < CLOCK_COUNT; ++id) {
            log(
                names[id],
                static_cast<int64_t>(clockCounts_[id]),
                totalClockCount,
                itemCounts_[id]
            );
        }
        cout << endl;

        cout << "zero calibration: " << zeroCalibration << endl;
        cout << "profiling overhead: "
            << (100.0 * totalZeroCalibration) / (totalClockCount + totalZeroCalibration) << "%" << endl;
        cout << "slow branch: " << (100.0 * slowBranchCount) / splitIterationCount << "%" << endl;

        for (int id = 0; id <= CLOCK_COUNT; ++id) {
            clockCounts_[id] = 0;
            itemCounts_[id] = 0;
            callCounts_[id] = 0;
        }
        slowBranchCount = 0;
        splitIterationCount = 0;
    }

#else

    inline void PUSH(ID) {}
    inline void POP(size_t) {}
    inline void PRINT() {}

#endif
}


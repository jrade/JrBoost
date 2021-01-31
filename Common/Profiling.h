#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


inline uint64_t clockCycleCount()
{
    __faststorefence();
    uint64_t t = __rdtsc();
    __faststorefence();
    return t;
}

//----------------------------------------------------------------------------------------------------------------------


namespace CLOCK {

    enum ID { MAIN, T_RANK, BT_TRAIN, ST_TRAIN, T1, T2, LCP_P, OMP_BARRIER, MEMORY, CLOCK_COUNT };

    inline void PUSHK(int i);
    inline void POP(size_t itemCount = 0);
    inline void PRINT();

    // implementation ..............................................................................

    inline uint64_t clocks[CLOCK_COUNT] = { 0 };
    inline uint64_t itemCounts[CLOCK_COUNT] = { 0 };

    const int clockStackSize = 256;
    inline int clockIndexStack[clockStackSize];
    inline int stackIndex = -1;

    inline void PUSH(int i)
    {
#pragma omp master
        {
            uint64_t t = clockCycleCount();
            if (stackIndex != -1) {
                int prevI = clockIndexStack[stackIndex];
                clocks[prevI] += t;
            }
            ++stackIndex;
            ASSERT(stackIndex != clockStackSize);
            clockIndexStack[stackIndex] = i;
            clocks[i] -= t;
        }
    }

    inline void POP(size_t itemCount)
    {
#pragma omp master
        {
            uint64_t t = clockCycleCount();
            ASSERT(stackIndex != -1);
            int i = clockIndexStack[stackIndex];
            clocks[i] += t;
            itemCounts[i] += itemCount;
            --stackIndex;
            if (stackIndex >= 0) {
                int prevI = clockIndexStack[stackIndex];
                clocks[prevI] -= t;
            }
        }
    }

    inline void log_(const char* s, int i)
    {
        float total = static_cast<float>(std::accumulate(std::begin(clocks), std::end(clocks), (uint64_t)0));

        cout << s;
        cout << std::setw(8) << std::fixed << std::setprecision(0) << 1e-6 * clocks[i];
        cout << "  ";
        cout << std::setw(4) << std::setprecision(1) << 100.0f * clocks[i] / total << "%";
        if (itemCounts[i] != 0) {
            cout << "  ";
            cout << std::setw(5) << static_cast<double>(clocks[i]) / itemCounts[i];
        }
        cout << endl;
    }

    inline void PRINT()
    {
        log_("main            ", MAIN);
        log_("  t-rank        ", T_RANK);
        log_("  train boost   ", BT_TRAIN);
        log_("    train stump ", ST_TRAIN);
        log_("      T1        ", T1);
        log_("      T2        ", T2);
        log_("  predict       ", LCP_P);
        log_("  OMP barrier   ", OMP_BARRIER);
        log_("  dyn. memory   ", MEMORY);

        cout << endl;

        std::fill(std::begin(clocks), std::end(clocks), 0);
        std::fill(std::begin(itemCounts), std::end(itemCounts), 0);
    }
}

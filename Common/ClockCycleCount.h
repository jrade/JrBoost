#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

inline uint64_t clockCycleCount() {
    _mm_lfence();  // wait for earlier instructions to retire before reading the clock
    uint64_t ccc =  __rdtsc();
    _mm_lfence();  // block later instructions until rdtsc retires
    return ccc;
}

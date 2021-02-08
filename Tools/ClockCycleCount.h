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


// Profiling by itself adds to the execution time that we are profiling.
// The fences increases the mean of the added time but decreases its variance
// and makes it easier to compensate for the added time

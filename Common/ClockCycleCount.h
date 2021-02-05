#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


inline int64_t clockCycleCount()
{
    __faststorefence();
    uint64_t t = __rdtsc();
    __faststorefence();
    return t;
}

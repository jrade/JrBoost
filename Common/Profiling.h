#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


inline void START_TIMER(uint64_t & t)
{
    __faststorefence();
    t -= __rdtsc();
    __faststorefence();
}

inline void STOP_TIMER(uint64_t& t)
{
    __faststorefence();
    t += __rdtsc();
    __faststorefence();
}

inline void SWITCH_TIMER(uint64_t& t1, uint64_t& t2)
{
    __faststorefence();
    const uint64_t t = __rdtsc();
    t1 += t;
    t2 -= t;
    __faststorefence();
}

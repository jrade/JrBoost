#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

inline uint64_t t__;
inline uint64_t t0__ = 0;
inline uint64_t t1__ = 0;
inline uint64_t t2__ = 0;
inline uint64_t t3__ = 0;

#define START_TIMER(T) t0__ = t1__ = t2__ = t3__ = 0; T -= __rdtsc()
#define STOP_TIMER(T) T += __rdtsc()
#define SWITCH_TIMER(T1, T2) t__ = __rdtsc(); T1 += t__; T2 -= t__

// Copyright (c) 2012, Johan Rade
// All rights reserved.
// <johan.rade@gmail.com>

#pragma once

#ifdef _MSC_VER				// Microsoft Visual Studio

#include <intrin.h>

inline int64_t clockCycleCount()
{
	return __rdtsc();
}

#elif defined(__GNUC__)		// GCC

inline int64_t clockCycleCount()
{
	int64_t a, d;
	__asm__ volatile ("rdtsc": "=a"(a), "=d"(d));
	return (d << 32) | a;
}

#else

#error Unknown compiler

#endif

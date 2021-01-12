#pragma once

// Standard library

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

// Eigen

#include <Eigen/Core>

using Eigen::ArrayXf;
using CRefXXf = Eigen::Ref<const Eigen::ArrayXXf>;

inline void assign(CRefXXf& a, const CRefXXf& b)
{
    a.~CRefXXf();
    new (&a) CRefXXf{ b };
}

inline Eigen::ArrayXXf dummyArrayXXf;

// Random number engines (by Arvid Gerstmann)

#include "AGRandom.h"

inline splitmix theRNE{ std::random_device{} };

// Profiling

#include "ClockCycleCount.h"

inline uint64_t t__;
inline uint64_t t0__ = 0;
inline uint64_t t1__ = 0; 
inline uint64_t t2__ = 0;
inline uint64_t t3__ = 0;
#define START_TIMER(T) T -= clockCycleCount()
#define STOP_TIMER(T) T += clockCycleCount()
#define SWITCH_TIMER(T1, T2) t__ = clockCycleCount(); T1 += t__; T2 -= t__

// Other

#include "Assert.h"
#include "FastAlgorithms.h"

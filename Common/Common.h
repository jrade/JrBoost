#pragma once

// Standard library

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using std::array;
using std::cout;
using std::endl;
using std::numeric_limits;
using std::runtime_error;
using std::string;
using std::vector;

// Eigen

#include <Eigen/Core>

using Eigen::ArrayXd;

using CRefXXf = Eigen::Ref<const Eigen::ArrayXXf>;

inline void assign(CRefXXf& a, const CRefXXf& b)
{
    a.~CRefXXf();
    new (&a) CRefXXf{ b };
}

// Random number engines (by Arvid Gerstmann)

#include "FastRandomNumbers.h"

// Other

#include "Assert.h"
#include "FastAlgorithms.h"
#include "Profiling.h"

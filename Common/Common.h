#pragma once

// Standard library

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using std::cout;
using std::endl;
using std::numeric_limits;
using std::optional;
using std::runtime_error;
using std::string;
using std::unique_ptr;
using std::vector;

// Eigen

#include <Eigen/Core>

using Eigen::ArrayXf;
using CRefXXf = Eigen::Ref<const Eigen::ArrayXXf>;

using ArrayXs = Eigen::Array<size_t, Eigen::Dynamic, 1>;

inline void assign(CRefXXf& a, const CRefXXf& b)
{
    a.~CRefXXf();
    new (&a) CRefXXf{ b };
}

inline Eigen::ArrayXXf dummyArrayXXf;

// Random number engines (by Arvid Gerstmann)

#include "AGRandom.h"

inline splitmix theRNE{ std::random_device{} };

// Other

#include "Assert.h"
#include "FastAlgorithms.h"
#include "Profiling.h"

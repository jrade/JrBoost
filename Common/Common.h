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

// Eigen arrays are by default column major
// NumPy arrays are by default row major

using PyArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXXf = Eigen::ArrayXXf;
using ArrayXd = Eigen::ArrayXd;
using ArrayXs = Eigen::Array<size_t, Eigen::Dynamic, 1>;

using PyRefXXd = Eigen::Ref<PyArrayXXd>;
using RefXXf = Eigen::Ref<ArrayXXf>;
using RefXd = Eigen::Ref<ArrayXd>;
using RefXs = Eigen::Ref<ArrayXs>;

using CPyRefXXd = Eigen::Ref<const PyArrayXXd>;
using CRefXXf = Eigen::Ref<const ArrayXXf>;
using CRefXd = Eigen::Ref<const ArrayXd>;
using CRefXs = Eigen::Ref<const ArrayXs>;


// Random number engines (by Arvid Gerstmann)

#include "FastRandomNumbers.h"


// Other

#include "Assert.h"
#include "FastAlgorithms.h"
#include "Profiling.h"

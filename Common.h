//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


// Standard library

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

using std::array;
using std::atomic;
using std::cout;
using std::endl;
using std::function;
using std::numeric_limits;
using std::optional;
using std::pair;
using std::runtime_error;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::unique_ptr;
using std::vector;


// OpenMP

#include <omp.h>


// Eigen

#define EIGEN_DONT_PARALLELIZE

#include <Eigen/Dense>

using ArrayXXf = Eigen::ArrayXXf;
using ArrayXd = Eigen::ArrayXd;
using ArrayXf = Eigen::ArrayXf;
using ArrayXs = Eigen::Array<size_t, Eigen::Dynamic, 1>;

using RefXXf = Eigen::Ref<ArrayXXf>;
using RefXd = Eigen::Ref<ArrayXd>;
using RefXs = Eigen::Ref<ArrayXs>;

using CRefXXf = Eigen::Ref<const ArrayXXf>;
using CRefXd = Eigen::Ref<const ArrayXd>;
using CRefXs = Eigen::Ref<const ArrayXs>;

using Array3d = Eigen::Array3d;


// Vector Class Library (by Agner Fog)

#define VCL_NAMESPACE vcl

#ifdef _MSC_VER
#   pragma warning( push )
#   pragma warning( disable : 4702 )
#endif

#include "vcl/vectorclass.h"
#include "vcl/vectormath_exp.h"

#ifdef _MSC_VER
#   pragma warning( pop )
#endif


// Tools

#include "JrBoostLib/Assert.h"
#include "JrBoostLib/Profile.h"

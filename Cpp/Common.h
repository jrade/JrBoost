//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


// Standard library

#include <algorithm>
#include <any>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <tuple>
#include <variant>
#include <vector>

using std::array;
using std::function;
using std::ifstream;
using std::istream;
using std::map;
using std::numeric_limits;
using std::ofstream;
using std::optional;
using std::ostream;
using std::pair;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::tuple;
using std::unique_ptr;
using std::vector;

inline const std::thread::id mainThreadId = std::this_thread::get_id();


// OpenMP

#include <omp.h>


// Eigen

#define EIGEN_DONT_PARALLELIZE

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable: 4127 5054)
#endif

#include <Eigen/Dense>

#ifdef _MSC_VER
#pragma warning( pop )
#endif

using ArrayXXd = Eigen::ArrayXXd;
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

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
inline const char* eigenVersion = STR(EIGEN_WORLD_VERSION) "." STR(EIGEN_MAJOR_VERSION) "." STR(EIGEN_MINOR_VERSION);


// Tools

#include "JrBoostLib/Assert.h"
#include "JrBoostLib/Profile.h"

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
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <tuple>
#include <variant>
#include <vector>

using std::array;
using std::cout;
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

using ArrayXXdc = Eigen::ArrayXXd;      // column major
using ArrayXXdr = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXXfc = Eigen::ArrayXXf;      // column major
using ArrayXXfr = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXd   = Eigen::ArrayXd;
using ArrayXf   = Eigen::ArrayXf;
using ArrayXs   = Eigen::Array<size_t, Eigen::Dynamic, 1>;

using RefXXdc = Eigen::Ref<ArrayXXdc>;
using RefXXdr = Eigen::Ref<ArrayXXdr>;
using RefXXfc = Eigen::Ref<ArrayXXfc>;
using RefXXfr = Eigen::Ref<ArrayXXfr>;
using RefXd   = Eigen::Ref<ArrayXd>;
using RefXf   = Eigen::Ref<ArrayXf>;
using RefXs   = Eigen::Ref<ArrayXs>;

using CRefXXdc = Eigen::Ref<const ArrayXXdc>;
using CRefXXdr = Eigen::Ref<const ArrayXXdr>;
using CRefXXfc = Eigen::Ref<const ArrayXXfc>;
using CRefXXfr = Eigen::Ref<const ArrayXXfr>;
using CRefXd   = Eigen::Ref<const ArrayXd>;
using CRefXf   = Eigen::Ref<const ArrayXf>;
using CRefXs   = Eigen::Ref<const ArrayXs>;


// Tools

#include "JrBoostLib/AGRandom.h"
#include "JrBoostLib/Assert.h"
#include "JrBoostLib/Profile.h"


// Global data

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
inline const char* theEigenVersion = STR(EIGEN_WORLD_VERSION) "." STR(EIGEN_MAJOR_VERSION) "." STR(EIGEN_MINOR_VERSION);

inline bool theParallelTree = true;
inline size_t theOuterThreadCount = 0;

inline const std::thread::id theMainThreadId = std::this_thread::get_id();

using RandomNumberEngine = splitmix;
inline thread_local RandomNumberEngine theRne((std::random_device()));

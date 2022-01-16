//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


// Standard library

#include <algorithm>
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
#include <sstream>
#include <stdexcept>
#include <string>
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


// OpenMP

#include <omp.h>


// Intel intrinsics

#include <immintrin.h>


// Eigen

#define EIGEN_DONT_PARALLELIZE

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)   // conditional expression is constant
#pragma warning(disable : 4805)   // '|': unsafe mix of type 'const bool' and type 'int' in operation
#endif                            // disabling 4805 only needed when compiling with AVX512

#include <Eigen/Dense>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using ArrayXXdc = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using ArrayXXdr = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXXfc = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using ArrayXXfr = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Array2Xdr = Eigen::Array<double, 2, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXd = Eigen::ArrayX<double>;
using ArrayXf = Eigen::ArrayX<float>;
using ArrayXs = Eigen::ArrayX<size_t>;
using ArrayXu8 = Eigen::ArrayX<uint8_t>;

using RefXXdc = Eigen::Ref<ArrayXXdc>;
using RefXXdr = Eigen::Ref<ArrayXXdr>;
using RefXXfc = Eigen::Ref<ArrayXXfc>;
using RefXXfr = Eigen::Ref<ArrayXXfr>;
using Ref2Xdr = Eigen::Ref<Array2Xdr>;
using RefXd = Eigen::Ref<ArrayXd>;
using RefXf = Eigen::Ref<ArrayXf>;
using RefXs = Eigen::Ref<ArrayXs>;
using RefXu8 = Eigen::Ref<ArrayXu8>;

using CRefXXdc = Eigen::Ref<const ArrayXXdc>;
using CRefXXdr = Eigen::Ref<const ArrayXXdr>;
using CRefXXfc = Eigen::Ref<const ArrayXXfc>;
using CRefXXfr = Eigen::Ref<const ArrayXXfr>;
using CRef2Xdr = Eigen::Ref<const Array2Xdr>;
using CRefXd = Eigen::Ref<const ArrayXd>;
using CRefXf = Eigen::Ref<const ArrayXf>;
using CRefXs = Eigen::Ref<const ArrayXs>;
using CRefXu8 = Eigen::Ref<const ArrayXu8>;

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
inline const char* theEigenVersion = STR(EIGEN_WORLD_VERSION) "." STR(EIGEN_MAJOR_VERSION) "." STR(EIGEN_MINOR_VERSION);


// Fast random number generators (by Arvid Gerstmann)

#include "3rdParty/random.h"


// pdqsort (by Orson Peters)

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)   // '=': conversion from 'size_t' to 'unsigned char', possible loss of data
#endif

#include "3rdParty/pdqsort.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif


// Miscellaneous

#include "JrBoostLib/Profile.h"
#include "JrBoostLib/Tools.h"


#ifdef _MSC_VER
#define USE_INTEL_INTRINSICS 1
#else
#define USE_INTEL_INTRINSICS 0
#endif

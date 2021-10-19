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
#pragma warning( push )
#pragma warning( disable: 4127 )    // conditional expression is constant
#pragma warning( disable: 4805 )    // '|': unsafe mix of type 'const bool' and type 'int' in operation                                 
#endif                              // disabling 4805 only needed when compiling with AVX512

#include <Eigen/Dense>

#ifdef _MSC_VER
#pragma warning( pop )
#endif

using ArrayXXdc = Eigen::ArrayXX<double>;      // column major
using ArrayXXdr = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXXfc = Eigen::ArrayXX<float>;      // column major
using ArrayXXfr = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXd   = Eigen::ArrayX<double>;
using ArrayXf   = Eigen::ArrayX<float>;
using ArrayXs   = Eigen::ArrayX<size_t>;

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


// Fast random number generators (by Arvid Gerstmann)

#include "3rdParty/random.h"


// pdqsort (by Orson Peters)

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable: 4267 )    // '=': conversion from 'size_t' to 'unsigned char', possible loss of data
#endif

#include "3rdParty/pdqsort.h"

#ifdef _MSC_VER
#pragma warning( pop )
#endif


// Tools

#include "JrBoostLib/Assert.h"
#include "JrBoostLib/Profile.h"


// Miscellaneous

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
inline const char* theEigenVersion = STR(EIGEN_WORLD_VERSION) "." STR(EIGEN_MAJOR_VERSION) "." STR(EIGEN_MINOR_VERSION);

inline bool globParallelTree = true;
inline size_t globOuterThreadCount = 0;

inline const std::thread::id theMainThreadId = std::this_thread::get_id();


using RandomNumberEngine = splitmix;

class InitializedRandomNumberEngine : public RandomNumberEngine
{
public:
    InitializedRandomNumberEngine() {
        std::random_device rd;
        seed(rd);
    }
};

inline thread_local InitializedRandomNumberEngine theRne;


template<class Iter, class Compare>
inline void fastSort(Iter begin, Iter end, Compare comp) {
    size_t n = end - begin;
    size_t ITEM_COUNT = static_cast<size_t>(std::round(n * std::log(n)));
    PROFILE::PUSH(PROFILE::SORT);
    pdqsort_branchless(begin, end, comp);
    PROFILE::POP(ITEM_COUNT);
}


#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable: 4324 )    // 'struct_name' : structure was padded due to __declspec(align())
#endif

template<typename T>
class alignas(std::hardware_destructive_interference_size) CacheLineAligned: public T {};

#ifdef _MSC_VER
#pragma warning( pop )
#endif

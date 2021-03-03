#pragma once


// Standard library

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include<functional>
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
#include <thread>
#include <tuple>
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
using std::tuple;
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


// Tools

#include "JrBoostLib/Assert.h"
#include "JrBoostLib/Profile.h"
#include "JrBoostLib/pdqsort.h"

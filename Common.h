#pragma once


// Standard library

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using std::array;
using std::cout;
using std::endl;
using std::numeric_limits;
using std::optional;
using std::pair;
using std::runtime_error;
using std::shared_ptr;
using std::string;
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
using ArrayXs = Eigen::Array<size_t, Eigen::Dynamic, 1>;

using RefXXf = Eigen::Ref<ArrayXXf>;
using RefXd = Eigen::Ref<ArrayXd>;
using RefXs = Eigen::Ref<ArrayXs>;

using CRefXXf = Eigen::Ref<const ArrayXXf>;
using CRefXd = Eigen::Ref<const ArrayXd>;
using CRefXs = Eigen::Ref<const ArrayXs>;

using ArrayXf = Eigen::ArrayXf;


// Tools

#include "Tools/Assert.h"
#include "Tools/Profile.h"




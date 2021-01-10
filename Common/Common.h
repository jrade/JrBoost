#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

#include <Eigen/Core>

using Eigen::ArrayXf;
using Eigen::ArrayXd;
using RefXXf = Eigen::Ref<Eigen::ArrayXXf>;

inline void assign(RefXXf& a, const RefXXf& b)
{
    a.~RefXXf();
    new (&a) RefXXf{ b };
}

inline Eigen::ArrayXXf dummyArrayXXf;

#include "AGRandom.h"
#include "Assert.h"
#include "ClockCycleCount.h"
#include "FastAlgorithms.h"

inline splitmix theRNE{ std::random_device{} };     // random number engine

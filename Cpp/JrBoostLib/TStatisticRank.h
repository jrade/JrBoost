//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "SortedIndices.h"


enum class TestDirection { Up, Down, Any };


inline ArrayXs tStatisticRank(
    Eigen::Ref<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> inData,
    CRefXs outData,
    optional<CRefXs> optSamples = optional<CRefXs>(),
    TestDirection testDirection = TestDirection::Any
)
{
    PROFILE::PUSH(PROFILE::T_RANK);

    size_t variableCount = inData.cols();
    ASSERT(outData.rows() == inData.rows());

    ArrayXs samples;
    size_t sampleCount;
    if (optSamples) {
        samples = *optSamples;
        sampleCount = samples.size();
    }
    else {
        sampleCount = inData.rows();
        samples.resize(sampleCount);
        for (size_t i = 0; i < sampleCount; ++i)
            samples(i) = i;
    }

    size_t n[2] = { 0, 0 };
    ArrayXf mean[2];
    mean[0] = ArrayXf::Constant(variableCount, 0.0f);
    mean[1] = ArrayXf::Constant(variableCount, 0.0f);
    for (int i = 0; i < sampleCount; ++i) {
        size_t j = samples[i];
        size_t k = outData(j);
        ++n[k];
        mean[k] += inData.row(j);
    }
    for (int k = 0; k < 2; ++k)
        mean[k] /= static_cast<float>(n[k]);

    ArrayXf SS[2];
    SS[0] = ArrayXf::Constant(variableCount, 0.0f);
    SS[1] = ArrayXf::Constant(variableCount, 0.0f);
    for (int i = 0; i < sampleCount; ++i) {
        size_t j = samples[i];
        size_t k = outData(j);
        SS[k] += (inData.row(j) - mean[k].transpose()).square();
    }

    // avoid calculating the square root (not needed since we only use the statistic to rank the variables)

    ArrayXf t2;

    switch (testDirection) {
    case TestDirection::Up:
        t2 = (mean[1] - mean[0]) * (mean[1] - mean[0]).abs();
        break;
    case TestDirection::Down:
        t2 = (mean[0] - mean[1]) * (mean[1] - mean[0]).abs();
        break;
    case TestDirection::Any:
        t2 = (mean[1] - mean[0]).square();
        break;
    }
      
    t2 /= (SS[0] + SS[1] + numeric_limits<float>::min());

    ArrayXs ind(variableCount);
    sortedIndices(
        &t2(0),
        &t2(0) + variableCount,
        &ind(0),
        [](auto x) { return -x; }
    );

    PROFILE::POP(sampleCount * variableCount);

    return ind;
}

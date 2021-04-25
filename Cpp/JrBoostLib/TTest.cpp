//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TTest.h"
#include "SortedIndices.h"


ArrayXf tStatistic(
    Eigen::Ref<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> inData,
    CRefXs outData,
    optional<CRefXs> optSamples
)
{
    size_t variableCount = inData.cols();
    ASSERT(outData.rows() == inData.rows());
    ASSERT((outData < 2).all());

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
    mean[0] = ArrayXf::Zero(variableCount);
    mean[1] = ArrayXf::Zero(variableCount);
    for (int i = 0; i < sampleCount; ++i) {
        size_t j = samples[i];
        size_t k = outData(j);
        ++n[k];
        mean[k] += inData.row(j);
    }

    if (n[0] == 0)
        throw runtime_error("Unable to do t-test: First group is empty");
    if (n[1] == 0)
        throw runtime_error("Unable to do t-test: Second group is empty");
    if (n[0] + n[1] == 2)
        throw runtime_error("Unable to do t-test: Zero degrees of freedom");

    for (int k = 0; k < 2; ++k)
        mean[k] /= static_cast<float>(n[k]);

    ArrayXf SS[2];
    SS[0] = ArrayXf::Zero(variableCount);
    SS[1] = ArrayXf::Zero(variableCount);
    for (int i = 0; i < sampleCount; ++i) {
        size_t j = samples[i];
        size_t k = outData(j);
        SS[k] += (inData.row(j) - mean[k].transpose()).square();
    }

    float a = sqrt(
        (n[0] + n[1] - 2.0f)
        /
        (1.0f / n[0] + 1.0f / n[1])
    );

    ArrayXf t =
        a
        *
        (mean[1] - mean[0])
        /
        (SS[0] + SS[1] + numeric_limits<float>::min()).sqrt();

    ASSERT((t < numeric_limits<float>::infinity()).all());
    ASSERT((t > -numeric_limits<float>::infinity()).all());

    return t;
}


ArrayXs tTestRank(
    Eigen::Ref<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> inData,
    CRefXs outData,
    optional<CRefXs> optSamples,
    TestDirection testDirection
)
{
    PROFILE::PUSH(PROFILE::T_RANK);
    size_t sampleCount = optSamples ? optSamples->size() : static_cast<size_t>(inData.rows());
    size_t variableCount = inData.cols();
    size_t ITEM_COUNT = sampleCount * variableCount;

    ArrayXf t = tStatistic(inData, outData, optSamples);

    switch (testDirection) {
    case TestDirection::Up:
        break;
    case TestDirection::Down:
        t = -t;
        break;
    case TestDirection::Any:
        t = t.abs();
        break;
    default:
        ASSERT(false);
    }

    ArrayXs ind(variableCount);
    sortedIndices(
        &t(0),
        &t(0) + variableCount,
        &ind(0),
        [](auto x) { return -x; }
    );

    PROFILE::POP(ITEM_COUNT);

    return ind;
}

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TTest.h"
#include "SortedIndices.h"


using AccScalar = float;
using DataArray = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using StatArray = Eigen::Array<AccScalar, 2, Eigen::Dynamic, Eigen::RowMajor>;


ArrayXf tStatistic(Eigen::Ref<const DataArray> inData, CRefXs outData, optional<CRefXs> optSamples)
{
    size_t variableCount = inData.cols();
    if (outData.rows() != inData.rows())
        throw std::invalid_argument("Indata and outdata have different numbers of samples.");
    if((outData > 1).any())
        throw std::invalid_argument("Outdata has values that are not 0 or 1.");

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

    ArrayXs n = { {0,0} };
    for (size_t i : samples) {
        size_t s = outData(i);
        ++n(s);
    }

    if (n(0) == 0)
        throw std::invalid_argument("Unable to do t-test: First group is empty");
    if (n(1) == 0)
        throw std::invalid_argument("Unable to do t-test: Second group is empty");
    if (n(0) + n(1) == 2)
        throw std::invalid_argument("Unable to do t-test: Zero degrees of freedom");

    ArrayXf t(variableCount);

    AccScalar a = sqrt(
        (n(0) + n(1) - AccScalar(2))
        /
        (AccScalar(1) / n(0) + AccScalar(1) / n(1))
    );

    const size_t blockWidth = 256;
    const size_t blockCount = (variableCount + blockWidth - 1) / blockWidth;
    ASSERT(blockCount == static_cast<int>(blockCount));

#pragma omp parallel
    {
        StatArray mean(2, blockWidth);
        StatArray ss(2, blockWidth);

#pragma omp for
        for (int v = 0; v < static_cast<int>(blockCount); ++v) {

            size_t j0 = v * blockWidth;
            size_t j1 = j0 + blockWidth;
            j1 = std::min(j1, variableCount);

            Eigen::Ref<const DataArray> inDataBlock(inData.block(0, j0, sampleCount, j1 - j0));
            Eigen::Ref<StatArray> meanBlock(mean.block(0, 0, 2, j1 - j0));
            Eigen::Ref<StatArray> ssBlock(ss.block(0, 0, 2, j1 - j0));
            Eigen::Ref<ArrayXf> tBlock(t.segment(j0, j1 - j0));

            meanBlock = AccScalar(0);
            for (size_t i: samples) {
                size_t s = outData(i);
                meanBlock.row(s) += inDataBlock.row(i).cast<AccScalar>();
            }
            meanBlock.colwise() /= n.cast<AccScalar>();

            ssBlock = AccScalar(0);
            for (size_t i : samples) {
                size_t s = outData(i);
                ssBlock.row(s) += (inDataBlock.row(i).cast<AccScalar>() - meanBlock.row(s)).square();
            }

            tBlock = (
                a * (meanBlock.row(1) - meanBlock.row(0))
                / (ssBlock.row(0) + ssBlock.row(1) + numeric_limits<AccScalar>::min()).sqrt()
            ).cast<float>();
        }
    }

    if (!(t.abs() < numeric_limits<float>::infinity()).all()) {
        if (!(inData.abs() < numeric_limits<float>::infinity()).all())
            throw std::invalid_argument("Indata has values that are infinity or NaN.");
        else
            throw std::overflow_error("Numerical overflow when calculating the t-statistic.");
    }

    return t;
}


ArrayXs tTestRank(
    Eigen::Ref<const DataArray> inData,
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

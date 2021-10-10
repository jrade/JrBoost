//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TTest.h"

#include "SortedIndices.h"


using AccType = double;
using StatArray = Eigen::Array<AccType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// AccType = double will give more accuracy
// With AccType = float the compiler ought to be able to SIMD vectorize the code, but MSVC 2019 does not


inline size_t divideRoundUp(size_t a, size_t b)
{
    return (a + b - 1) / b;
}


ArrayXf tStatistic(CRefXXfr inData, CRefXs outData, optional<vector<size_t>> optSamples)
{        
    PROFILE::PUSH(PROFILE::T_RANK);

    const size_t variableCount = inData.cols();
    if (outData.rows() != inData.rows())
        throw std::invalid_argument("Indata and outdata have different numbers of samples.");
    if ((outData > 1).any())
        throw std::invalid_argument("Outdata has values that are not 0 or 1.");

    vector<size_t> samples;
    size_t sampleCount;
    if (optSamples) {
        samples = std::move(*optSamples);
        sampleCount = size(samples);
    }
    else {
        sampleCount = inData.rows();
        samples.resize(sampleCount);
        for (size_t i = 0; i < sampleCount; ++i)
            samples[i] = i;
    }

    if (sampleCount < 3)
        throw std::invalid_argument("Unable to do t-test: There must be at least three samples.");


    ArrayXs n = { {0, 0} };
    for (size_t i : samples) {
        size_t s = outData(i);
        ++n(s);
    }
    // n(0) + n(1) = sampleCount

    if (n(0) == 0)
        throw std::invalid_argument("Unable to do t-test: First group is empty.");
    if (n(1) == 0)
        throw std::invalid_argument("Unable to do t-test: Second group is empty.");

    ArrayXf t(variableCount);

    const AccType a = std::sqrt(
        (sampleCount - AccType(2))
        /
        (AccType(1) / n(0) + AccType(1) / n(1))
    );

    const AccType c = sampleCount * numeric_limits<AccType>::epsilon();

    // one block per thread...
    size_t blockCount = omp_get_max_threads();
    size_t blockWidth = divideRoundUp(variableCount, blockCount);
    // ... but avoid too small blocks
    const size_t minBlockWidth = 256;
    blockWidth = minBlockWidth * divideRoundUp(blockWidth, minBlockWidth);
    blockCount = divideRoundUp(variableCount, blockWidth);

#pragma omp parallel num_threads(static_cast<int>(blockCount))
    {
        ASSERT(static_cast<size_t>(omp_get_num_threads()) == blockCount);

        StatArray mean(2, blockWidth);
        StatArray totalMean(1, blockWidth);
        StatArray ss(2, blockWidth);

        const size_t k = omp_get_thread_num();      // block index
        const size_t j0 = k * blockWidth;
        const size_t j1 = std::min((k + 1) * blockWidth, variableCount);

        CRefXXfr inDataBlock(inData.block(0, j0, sampleCount, j1 - j0));
        Eigen::Ref<StatArray> meanBlock(mean.block(0, 0, 2, j1 - j0));
        Eigen::Ref<StatArray> totalMeanBlock(totalMean.block(0, 0, 1, j1 - j0));
        Eigen::Ref<StatArray> ssBlock(ss.block(0, 0, 2, j1 - j0));
        RefXf tBlock(t.segment(j0, j1 - j0));

        meanBlock = AccType(0);
        for (size_t i: samples) {
            size_t s = outData(i);
            meanBlock.row(s) += inDataBlock.row(i).cast<AccType>();
        }
        meanBlock.colwise() /= n.cast<AccType>();

        ssBlock = AccType(0);
        for (size_t i : samples) {
            size_t s = outData(i);
            ssBlock.row(s) += (inDataBlock.row(i).cast<AccType>() - meanBlock.row(s)).square();
        }

        totalMeanBlock = (n(0) * meanBlock.row(0) + n(1) * meanBlock.row(1)) / sampleCount;

        tBlock = (
            a
            *
            (meanBlock.row(1) - meanBlock.row(0))
            /
            (
                ssBlock.row(0) + ssBlock.row(1)
                +
                c * totalMeanBlock.square()             // fudge term
            ).sqrt()
        ).cast<float>();
    }

    if (!(t.abs() < numeric_limits<float>::infinity()).all()) {
        if (!(inData.abs() < numeric_limits<float>::infinity()).all())
            throw std::invalid_argument("Indata has values that are infinity or NaN.");
        else
            // The fudge term should usually prevent this from happening
            throw std::overflow_error("Numerical overflow when calculating the t-statistic.");
    }

    const size_t ITEM_COUNT = sampleCount * blockWidth;
    PROFILE::POP(ITEM_COUNT);

    return t;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXs tTestRank(CRefXXfr inData, CRefXs outData, optional<vector<size_t>> optSamples, TestDirection testDirection)
{
    const size_t variableCount = inData.cols();

    ArrayXf t = tStatistic(inData, outData, std::move(optSamples));

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

    return ind;
}

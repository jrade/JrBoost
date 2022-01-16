//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "TTest.h"

#include "OmpParallel.h"


ArrayXf tStatistic(CRefXXfr inData, CRefXu8 outData, optional<CRefXs> samples)
{
    size_t ITEM_COUNT = 0;
    ScopedProfiler sp(PROFILE::T_RANK, &ITEM_COUNT);

    if (outData.rows() != inData.rows())
        throw std::invalid_argument("Indata and outdata have different numbers of samples.");
    if ((outData > 1).any())
        throw std::invalid_argument("Outdata has values that are not 0 or 1.");

    const size_t sampleCount = samples ? samples->rows() : inData.rows();

    if (sampleCount < 3)
        throw std::invalid_argument("Unable to do t-test: There must be at least three samples.");

    ArrayXs n = {{0, 0}};
    if (samples) {
        for (size_t i : *samples) {
            uint8_t k = outData(i);
            ++n(k);
        }
    }
    else {
        for (size_t i = 0; i != sampleCount; ++i) {
            uint8_t k = outData(i);
            ++n(k);
        }
    }

    // n(0) + n(1) = sampleCount

    if (n(0) == 0)
        throw std::invalid_argument("Unable to do t-test: First group is empty.");
    if (n(1) == 0)
        throw std::invalid_argument("Unable to do t-test: Second group is empty.");

    const size_t variableCount = inData.cols();
    ArrayXf t(variableCount);
    const double a = std::sqrt((sampleCount - 2.0) / (1.0 / n(0) + 1.0 / n(1)));

    // one block per thread...
    size_t blockCount = omp_get_max_threads();
    size_t blockWidth = ::divideRoundUp(variableCount, blockCount);
    // ... but avoid too small blocks
    const size_t minBlockWidth = 256;
    blockWidth = minBlockWidth * ::divideRoundUp(blockWidth, minBlockWidth);
    blockCount = ::divideRoundUp(variableCount, blockWidth);

    BEGIN_OMP_PARALLEL(blockCount)
    {
        Array2Xdr mean(2, blockWidth);
        Array2Xdr ss(2, blockWidth);

        const size_t blockIndex = omp_get_thread_num();
        const size_t j0 = blockIndex * blockWidth;
        const size_t j1 = std::min((blockIndex + 1) * blockWidth, variableCount);

        CRefXXfr inDataBlock(inData.block(0, j0, sampleCount, j1 - j0));
        Ref2Xdr meanBlock(mean.block(0, 0, 2, j1 - j0));
        Ref2Xdr ssBlock(ss.block(0, 0, 2, j1 - j0));
        RefXf tBlock(t.segment(j0, j1 - j0));

        meanBlock = 0.0;
        if (samples) {
            for (size_t i : *samples) {
                uint8_t k = outData(i);
                meanBlock.row(k) += inDataBlock.row(i).cast<double>();
            }
        }
        else {
            for (size_t i = 0; i != sampleCount; ++i) {
                uint8_t k = outData(i);
                meanBlock.row(k) += inDataBlock.row(i).cast<double>();
            }
        }
        meanBlock.colwise() /= n.cast<double>();

        ssBlock = 0.0;
        if (samples) {
            for (size_t i : *samples) {
                uint8_t k = outData(i);
                ssBlock.row(k) += (inDataBlock.row(i).cast<double>() - meanBlock.row(k)).square();
            }
        }
        else {
            for (size_t i = 0; i != sampleCount; ++i) {
                uint8_t k = outData(i);
                ssBlock.row(k) += (inDataBlock.row(i).cast<double>() - meanBlock.row(k)).square();
            }
        }

        tBlock = (a * (meanBlock.row(1) - meanBlock.row(0))
                  / (ssBlock.row(0) + ssBlock.row(1) + numeric_limits<double>::min()).sqrt())
                     .cast<float>();
    }
    END_OMP_PARALLEL

    if (t.isNaN().any()) {
        ASSERT(!inData.isFinite().all());
        throw std::invalid_argument("Indata has values that are infinity or NaN.");
    }

    ITEM_COUNT = sampleCount * blockWidth;   // used by ScopedProfiier destructor
    return t;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXs tTestRank(CRefXXfr inData, CRefXu8 outData, optional<CRefXs> samples, TestDirection testDirection)
{
    const size_t variableCount = inData.cols();

    ArrayXf t = tStatistic(inData, outData, samples);

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

    sortedIndices(std::data(t), std::data(t) + variableCount, std::data(ind), [](auto x) { return -x; });

    return ind;
}

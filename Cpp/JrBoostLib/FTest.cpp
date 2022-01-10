//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "FTest.h"


ArrayXf fStatistic(CRefXXfr inData, CRefXu8 outData, optional<CRefXs> samples)
{
    PROFILE::PUSH(PROFILE::F_RANK);

    const size_t variableCount = inData.cols();
    if (outData.rows() != inData.rows())
        throw std::invalid_argument("Indata and outdata have different numbers of samples.");

    const size_t sampleCount = samples ? samples->rows() : inData.rows();

    size_t groupCount = 0;
    if (samples) {
        for (size_t i : *samples) {
            uint8_t k = outData(i);
            groupCount = std::max(groupCount, static_cast<size_t>(k) + 1);
        }
    }
    else {
        for (size_t i = 0; i != sampleCount; ++i) {
            uint8_t k = outData(i);
            groupCount = std::max(groupCount, static_cast<size_t>(k) + 1);
        }
    }

    if (groupCount < 2) {
        PROFILE::POP();
        throw std::invalid_argument("Unable to do F-test: There must be at least two groups.");
    }
    if (sampleCount <= groupCount) {
        PROFILE::POP();
        throw std::invalid_argument("Unable to do F-test: There must be more samples than groups.");
    }

    ArrayXs n = ArrayXs::Zero(groupCount);   // sample count by group
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

    for (size_t k = 0; k != groupCount; ++k) {
        if (n(k) == 0)
            throw std::invalid_argument(
                "Unable to do F-test: The group with index " + std::to_string(k) + " is empty.");
    }


    ArrayXf f(variableCount);
    const double a = (sampleCount - groupCount) / (groupCount - 1.0);

    // one block per thread...
    size_t blockCount = omp_get_max_threads();
    size_t blockWidth = divideRoundUp(variableCount, blockCount);
    // ... but avoid too small blocks
    const size_t minBlockWidth = 256;
    blockWidth = minBlockWidth * divideRoundUp(blockWidth, minBlockWidth);
    blockCount = divideRoundUp(variableCount, blockWidth);

#pragma omp parallel num_threads(static_cast <int>(blockCount))
    {
        ASSERT(static_cast<size_t>(omp_get_num_threads()) == blockCount);

        ArrayXXdr mean(groupCount, blockWidth);
        ArrayXXdr totalMean(1, blockWidth);
        ArrayXXdr ss(groupCount, blockWidth);

        const size_t blockIndex = omp_get_thread_num();   // block index
        const size_t j0 = blockIndex * blockWidth;
        const size_t j1 = std::min((blockIndex + 1) * blockWidth, variableCount);

        CRefXXfr inDataBlock(inData.block(0, j0, sampleCount, j1 - j0));
        RefXXdr meanBlock(mean.block(0, 0, groupCount, j1 - j0));
        RefXXdr totalMeanBlock(totalMean.block(0, 0, 1, j1 - j0));
        RefXXdr ssBlock(ss.block(0, 0, groupCount, j1 - j0));
        RefXf fBlock(f.segment(j0, j1 - j0));

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
        totalMeanBlock = meanBlock.colwise().sum() / sampleCount;
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

        fBlock
            = (a * ((meanBlock.rowwise() - totalMeanBlock.row(0)).square().colwise() * n.cast<double>()).colwise().sum()
               / (ssBlock.colwise().sum() + numeric_limits<double>::min()))
                  .cast<float>();
    }

    if (f.isNaN().any()) {
        ASSERT(!inData.isFinite().all());
        throw std::invalid_argument("Indata has values that are infinity or NaN.");
    }

    const size_t ITEM_COUNT = sampleCount * blockWidth;
    PROFILE::POP(ITEM_COUNT);

    return f;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXs fTestRank(CRefXXfr inData, CRefXu8 outData, optional<CRefXs> samples)
{
    const size_t variableCount = inData.cols();
    const ArrayXf f = fStatistic(inData, outData, samples);
    ArrayXs ind(variableCount);

    sortedIndices(std::data(f), std::data(f) + variableCount, std::data(ind), [](auto x) { return -x; });

    return ind;
}

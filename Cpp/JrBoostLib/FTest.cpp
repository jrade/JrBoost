//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "FTest.h"

#include "SortedIndices.h"


using AccType = double;
using StatArray = Eigen::Array<AccType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// AccType = double will give more accuracy
// With AccType = float the compiler ought to be able to SIMD vectorize the code, but MSVC 2019 does not


inline size_t divideRoundUp(size_t a, size_t b)
{
    return (a + b - 1) / b;
}


ArrayXf fStatistic(CRefXXfr inData, CRefXs outData, optional<vector<size_t>> optSamples)
{        
    const size_t variableCount = inData.cols();
    if (outData.rows() != inData.rows())
        throw std::invalid_argument("Indata and outdata have different numbers of samples.");

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

    size_t groupCount = 0;
    for (size_t i : samples)
        groupCount = std::max(groupCount, outData(i));
    groupCount += 1;

    if (groupCount < 2)
        throw std::invalid_argument("Unable to do F-test: There must be at least two groups.");
    if (sampleCount <= groupCount)
        throw std::invalid_argument("Unable to do F-test: There must be more samples than groups.");

    ArrayXs n = ArrayXs::Zero(groupCount);
    for (size_t i : samples) {
        size_t s = outData(i);
        ++n(s);
    }
    // n.sum() == sampleCount

    if (n.minCoeff() == 0)
        for (size_t i = 0; i < groupCount; ++i)
            if (n(i) == 0)
                throw std::invalid_argument(
                    "Unable to do F-test: The group with index " + std::to_string(i) + " is empty."
                );

  
    ArrayXf f(variableCount);
    const AccType a = static_cast<AccType>(sampleCount - groupCount) / static_cast<AccType>(groupCount - 1);
    const AccType c = static_cast<AccType>(sampleCount) * numeric_limits<AccType>::epsilon();

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

        StatArray mean(groupCount, blockWidth);
        StatArray totalMean(1, blockWidth);
        StatArray ss(groupCount, blockWidth);

        const size_t k = omp_get_thread_num();      // block index
        const size_t j0 = k * blockWidth;
        const size_t j1 = std::min((k + 1) * blockWidth, variableCount);

        CRefXXfr inDataBlock(inData.block(0, j0, sampleCount, j1 - j0));
        Eigen::Ref<StatArray> meanBlock(mean.block(0, 0, groupCount, j1 - j0));
        Eigen::Ref<StatArray> totalMeanBlock(totalMean.block(0, 0, 1, j1 - j0));
        Eigen::Ref<StatArray> ssBlock(ss.block(0, 0, groupCount, j1 - j0));
        RefXf fBlock(f.segment(j0, j1 - j0));

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

        totalMeanBlock = (meanBlock.colwise() * n.cast<AccType>()).colwise().sum() / sampleCount;

        fBlock = (
            a
            *
            (
                (meanBlock.rowwise() - totalMeanBlock.row(0)).square().colwise()
                *
                n.cast<AccType>()
            ).colwise().sum()
            /
            (
                ssBlock.colwise().sum()
                +
                c * totalMeanBlock.square()     // fudge term
            )
        ).cast<float>();
    }

    if (!(f.abs() < numeric_limits<float>::infinity()).all()) {
        if (!(inData.abs() < numeric_limits<float>::infinity()).all())
            throw std::invalid_argument("Indata has values that are infinity or NaN.");
        else
            // The fudge term should usually prevent this from happening
            throw std::overflow_error("Numerical overflow when calculating the F-statistic.");
    }

    return f;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXs fTestRank(CRefXXfr inData, CRefXs outData, optional<vector<size_t>> optSamples)
{
    const size_t variableCount = inData.cols();
    const ArrayXf f = fStatistic(inData, outData, std::move(optSamples));
    ArrayXs ind(variableCount);

    sortedIndices(
        &f(0),
        &f(0) + variableCount,
        &ind(0),
        [](auto x) { return -x; }
    );

    return ind;
}

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "Submatrix.h"


ArrayXXfc selectRows(CRefXXfc inData, const vector<size_t>& samples)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    //const size_t inSamplesCount = static_cast<size_t>(inData.rows());
    const size_t outSampleCount = size(samples);
    const size_t variableCount = static_cast<size_t>(inData.cols());

    ArrayXXfc outData(outSampleCount, variableCount);
    if (outSampleCount == 0 || variableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), variableCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t jStart = variableCount * threadIndex / threadCount;
        const size_t jStop = variableCount * (threadIndex + 1) / threadCount;

        for (size_t j = jStart; j != jStop; ++j) {
            const float* pInDataColJ = std::data(inData.col(j));
            float* pOutDataColJ = &outData(0, j);
            for (size_t outI = 0; outI != outSampleCount; ++outI) {
                const size_t inI = samples[outI];
                pOutDataColJ[outI] = pInDataColJ[inI];
            }
        }
    }

    PROFILE::POP(outSampleCount * variableCount);

    return outData;
}


ArrayXXfr selectRows(CRefXXfr inData, const vector<size_t>& samples)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    //const size_t inSampleCount = static_cast<size_t>(inData.rows());
    const size_t outSampleCount = size(samples);
    const size_t variableCount = static_cast<size_t>(inData.cols());

    ArrayXXfr outData(outSampleCount, variableCount);
    if (outSampleCount == 0 || variableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), outSampleCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t outIStart = outSampleCount * threadIndex / threadCount;
        const size_t outIStop = outSampleCount * (threadIndex + 1) / threadCount;

        for (size_t outI = outIStart; outI != outIStop; ++outI) {
            size_t inI = samples[outI];
            const float* pInDataRowI = std::data(inData.row(inI));
            float* pOutDataRowI = &outData(outI, 0);
            std::copy(pInDataRowI, pInDataRowI + variableCount, pOutDataRowI);
        }
    }

    PROFILE::POP(outSampleCount * variableCount);

    return outData;
}


ArrayXXdc selectRows(CRefXXdc inData, const vector<size_t>& samples)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    //const size_t inSamplesCount = static_cast<size_t>(inData.rows());
    const size_t outSampleCount = size(samples);
    const size_t variableCount = static_cast<size_t>(inData.cols());

    ArrayXXdc outData(outSampleCount, variableCount);
    if (outSampleCount == 0 || variableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), variableCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t jStart = variableCount * threadIndex / threadCount;
        const size_t jStop = variableCount * (threadIndex + 1) / threadCount;

        for (size_t j = jStart; j != jStop; ++j) {
            const double* pInDataColJ = std::data(inData.col(j));
            double* pOutDataColJ = &outData(0, j);
            for (size_t outI = 0; outI != outSampleCount; ++outI) {
                const size_t inI = samples[outI];
                pOutDataColJ[outI] = pInDataColJ[inI];
            }
        }
    }

    PROFILE::POP(outSampleCount * variableCount);

    return outData;
}


ArrayXXdr selectRows(CRefXXdr inData, const vector<size_t>& samples)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    //const size_t inSampleCount = static_cast<size_t>(inData.rows());
    const size_t outSampleCount = size(samples);
    const size_t variableCount = static_cast<size_t>(inData.cols());

    ArrayXXdr outData(outSampleCount, variableCount);
    if (outSampleCount == 0 || variableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), outSampleCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t outIStart = outSampleCount * threadIndex / threadCount;
        const size_t outIStop = outSampleCount * (threadIndex + 1) / threadCount;

        for (size_t outI = outIStart; outI != outIStop; ++outI) {
            size_t inI = samples[outI];
            const double* pInDataRowI = std::data(inData.row(inI));
            double* pOutDataRowI = &outData(outI, 0);
            std::copy(pInDataRowI, pInDataRowI + variableCount, pOutDataRowI);
        }
    }

    PROFILE::POP(outSampleCount * variableCount);

    return outData;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXXfc selectColumns(CRefXXfc inData, const vector<size_t>& variables)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    const size_t sampleCount = static_cast<size_t>(inData.rows());
    //const size_t inVariableCount = static_cast<size_t>(inData.cols());
    const size_t outVariableCount = size(variables);

    ArrayXXfc outData(sampleCount, outVariableCount);
    if (sampleCount == 0 || outVariableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), outVariableCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t outJStart = outVariableCount * threadIndex / threadCount;
        const size_t outJStop = outVariableCount * (threadIndex + 1) / threadCount;

        for (size_t outJ = outJStart; outJ != outJStop; ++outJ) {
            size_t inJ = variables[outJ];
            const float* pInDataColJ = std::data(inData.col(inJ));
            float* pOutDataColJ = &outData(0, outJ);
            std::copy(pInDataColJ, pInDataColJ + sampleCount, pOutDataColJ);
        }
    }

    PROFILE::POP(sampleCount * outVariableCount);

    return outData;
}


ArrayXXfr selectColumns(CRefXXfr inData, const vector<size_t>& variables)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    const size_t sampleCount = static_cast<size_t>(inData.rows());
    //const size_t inVariableCount = static_cast<size_t>(inData.cols());
    const size_t outVariableCount = size(variables);

    ArrayXXfr outData(sampleCount, outVariableCount);
    if (sampleCount == 0 || outVariableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), sampleCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t iStart = sampleCount * threadIndex / threadCount;
        const size_t iStop = sampleCount * (threadIndex + 1) / threadCount;

        for (size_t i = iStart; i != iStop; ++i) {
            const float* pInDataRowI = std::data(inData.row(i));
            float* pOutDataRowI = &outData(i, 0);
            for (size_t outJ = 0; outJ != outVariableCount; ++outJ) {
                const size_t inJ = variables[outJ];
                pOutDataRowI[outJ] = pInDataRowI[inJ];
            }
        }
    }

    PROFILE::POP(sampleCount * outVariableCount);

    return outData;
}


ArrayXXdc selectColumns(CRefXXdc inData, const vector<size_t>& variables)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    const size_t sampleCount = static_cast<size_t>(inData.rows());
    //const size_t inVariableCount = static_cast<size_t>(inData.cols());
    const size_t outVariableCount = size(variables);

    ArrayXXdc outData(sampleCount, outVariableCount);
    if (sampleCount == 0 || outVariableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), outVariableCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t outJStart = outVariableCount * threadIndex / threadCount;
        const size_t outJStop = outVariableCount * (threadIndex + 1) / threadCount;

        for (size_t outJ = outJStart; outJ != outJStop; ++outJ) {
            size_t inJ = variables[outJ];
            const double* pInDataColJ = std::data(inData.col(inJ));
            double* pOutDataColJ = &outData(0, outJ);
            std::copy(pInDataColJ, pInDataColJ + sampleCount, pOutDataColJ);
        }
    }

    PROFILE::POP(sampleCount * outVariableCount);

    return outData;
}


ArrayXXdr selectColumns(CRefXXdr inData, const vector<size_t>& variables)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    const size_t sampleCount = static_cast<size_t>(inData.rows());
    //const size_t inVariableCount = static_cast<size_t>(inData.cols());
    const size_t outVariableCount = size(variables);

    ArrayXXdr outData(sampleCount, outVariableCount);
    if (sampleCount == 0 || outVariableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), sampleCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t iStart = sampleCount * threadIndex / threadCount;
        const size_t iStop = sampleCount * (threadIndex + 1) / threadCount;

        for (size_t i = iStart; i != iStop; ++i) {
            const double* pInDataRowI = std::data(inData.row(i));
            double* pOutDataRowI = &outData(i, 0);
            for (size_t outJ = 0; outJ != outVariableCount; ++outJ) {
                const size_t inJ = variables[outJ];
                pOutDataRowI[outJ] = pInDataRowI[inJ];
            }
        }
    }

    PROFILE::POP(sampleCount * outVariableCount);

    return outData;
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXXfc select(CRefXXfc inData, const vector<size_t>& samples, const vector<size_t>& variables)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    //const size_t inSampleCount = static_cast<size_t>(inData.rows());
    const size_t outSampleCount = size(samples);
    //const size_t inVariableCount = static_cast<size_t>(inData.cols());
    const size_t outVariableCount = size(variables);

    ArrayXXfc outData(outSampleCount, outVariableCount);
    if (outSampleCount == 0 || outVariableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), outVariableCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t outJStart = outVariableCount * threadIndex / threadCount;
        const size_t outJStop = outVariableCount * (threadIndex + 1) / threadCount;

        for (size_t outJ = outJStart; outJ != outJStop; ++outJ) {
            size_t inJ = variables[outJ];
            const float* pInDataColJ = std::data(inData.col(inJ));
            float* pOutDataColJ = &outData(0, outJ);
            for (size_t outI = 0; outI != outSampleCount; ++outI) {
                const size_t inI = samples[outI];
                pOutDataColJ[outI] = pInDataColJ[inI];
            }
        }
    }

    PROFILE::POP(outSampleCount * outVariableCount);

    return outData;
}


ArrayXXfr select(CRefXXfr inData, const vector<size_t>& samples, const vector<size_t>& variables)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    //const size_t inSampleCount = static_cast<size_t>(inData.rows());
    const size_t outSampleCount = size(samples);
    //const size_t inVariableCount = static_cast<size_t>(inData.cols());
    const size_t outVariableCount = size(variables);

    ArrayXXfr outData(outSampleCount, outVariableCount);
    if (outSampleCount == 0 || outVariableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), outSampleCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t outIStart = outSampleCount * threadIndex / threadCount;
        const size_t outIStop = outSampleCount * (threadIndex + 1) / threadCount;

        for (size_t outI = outIStart; outI != outIStop; ++outI) {
            size_t inI = samples[outI];
            const float* pInDataRowI = std::data(inData.row(inI));
            float* pOutDataRowI = &outData(outI, 0);
            for (size_t outJ = 0; outJ != outVariableCount; ++outJ) {
                const size_t inJ = variables[outJ];
                pOutDataRowI[outJ] = pInDataRowI[inJ];
            }
        }
    }

    PROFILE::POP(outSampleCount * outVariableCount);

    return outData;
}


ArrayXXdc select(CRefXXdc inData, const vector<size_t>& samples, const vector<size_t>& variables)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    //const size_t inSampleCount = static_cast<size_t>(inData.rows());
    const size_t outSampleCount = size(samples);
    //const size_t inVariableCount = static_cast<size_t>(inData.cols());
    const size_t outVariableCount = size(variables);

    ArrayXXdc outData(outSampleCount, outVariableCount);
    if (outSampleCount == 0 || outVariableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), outVariableCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t outJStart = outVariableCount * threadIndex / threadCount;
        const size_t outJStop = outVariableCount * (threadIndex + 1) / threadCount;

        for (size_t outJ = outJStart; outJ != outJStop; ++outJ) {
            size_t inJ = variables[outJ];
            const double* pInDataColJ = std::data(inData.col(inJ));
            double* pOutDataColJ = &outData(0, outJ);
            for (size_t outI = 0; outI != outSampleCount; ++outI) {
                const size_t inI = samples[outI];
                pOutDataColJ[outI] = pInDataColJ[inI];
            }
        }
    }

    PROFILE::POP(outSampleCount * outVariableCount);

    return outData;
}


ArrayXXdr select(CRefXXdr inData, const vector<size_t>& samples, const vector<size_t>& variables)
{
    PROFILE::PUSH(PROFILE::SUBMATRIX);

    //const size_t inSampleCount = static_cast<size_t>(inData.rows());
    const size_t outSampleCount = size(samples);
    //const size_t inVariableCount = static_cast<size_t>(inData.cols());
    const size_t outVariableCount = size(variables);

    ArrayXXdr outData(outSampleCount, outVariableCount);
    if (outSampleCount == 0 || outVariableCount == 0) return outData;

    size_t threadCount = std::min<size_t>(omp_get_max_threads(), outSampleCount);
#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t outIStart = outSampleCount * threadIndex / threadCount;
        const size_t outIStop = outSampleCount * (threadIndex + 1) / threadCount;

        for (size_t outI = outIStart; outI != outIStop; ++outI) {
            size_t inI = samples[outI];
            const double* pInDataRowI = std::data(inData.row(inI));
            double* pOutDataRowI = &outData(outI, 0);
            for (size_t outJ = 0; outJ != outVariableCount; ++outJ) {
                const size_t inJ = variables[outJ];
                pOutDataRowI[outJ] = pInDataRowI[inJ];
            }
        }
    }

    PROFILE::POP(outSampleCount * outVariableCount);

    return outData;
}

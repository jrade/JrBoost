//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImplBase.h"


size_t TreeTrainerImplBase::bufferSize()
{
    size_t n = 0;
#pragma omp parallel reduction(+: n)
    {
        n += byteCount_(threadLocalData0_.usedVariables);
        n += byteCount_(threadLocalData0_.tree);
        for (const auto& layer : threadLocalData0_.tree)
            n += byteCount_(layer);

        n += bufferSizeImpl_<uint8_t>();
        n += bufferSizeImpl_<uint16_t>();
        n += bufferSizeImpl_<uint32_t>();
        n += bufferSizeImpl_<uint64_t>();
    }
    return n;
}

template<typename T>
size_t TreeTrainerImplBase::bufferSizeImpl_()
{
    size_t n = 0;

    n += byteCount_(threadLocalData1_<T>.orderedSamplesByVariable);
    for (const auto& orderedSamples : threadLocalData1_<T>.orderedSamplesByVariable)
        n += byteCount_(orderedSamples);
    n += byteCount_(threadLocalData1_<T>.sampleBuffer);
    n += byteCount_(threadLocalData1_<T>.samplePointerBuffer);
    n += byteCount_(threadLocalData1_<T>.treeNodeTrainers);

    n += byteCount_(threadLocalData2_<T>.sampleStatus);

    return n;
}

template<typename T>
size_t TreeTrainerImplBase::byteCount_(const T& t)
{
    return t.capacity() * sizeof(typename T::value_type);
}

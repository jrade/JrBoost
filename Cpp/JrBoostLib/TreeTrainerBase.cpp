//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerBase.h"


size_t TreeTrainerBase::bufferSize()
{
    size_t n = 0;
#pragma omp parallel reduction(+: n)
    {
#if PACKED_DATA
        n += bufferSize0_(threadLocalData0_.wyPacks);
#endif
        n += bufferSizeImpl_(threadLocalData0_.usedVariables);
        n += bufferSizeImpl_(threadLocalData0_.tree);

        n += bufferSizeImpl_<uint8_t>();
        n += bufferSizeImpl_<uint16_t>();
        n += bufferSizeImpl_<uint32_t>();
        n += bufferSizeImpl_<uint64_t>();
    }
    return n;
}

template<typename T>
size_t TreeTrainerBase::bufferSizeImpl_()
{
    size_t n = 0;

    n += bufferSizeImpl_(threadLocalData1_<T>.orderedSamplesByVariable);
    n += bufferSizeImpl_(threadLocalData1_<T>.sampleBuffer);
    n += bufferSizeImpl_(threadLocalData1_<T>.samplePointerBuffer);
    n += bufferSizeImpl_(threadLocalData1_<T>.treeNodeTrainers);

    n += bufferSizeImpl_(threadLocalData2_<T>.sampleStatus);

    return n;
}

template<typename T>
size_t TreeTrainerBase::bufferSizeImpl_(const vector<T>& t)
{
    return t.capacity() * sizeof(T);
}

template<typename T>
size_t TreeTrainerBase::bufferSizeImpl_(const vector<vector<T>>& t)
{
    size_t n = t.capacity() * sizeof(vector<T>);
    for (const auto& u : t)
        n += u.capacity() * sizeof(T);
    return n;
}

//......................................................................................................................

void TreeTrainerBase::freeBuffers()
{
#pragma omp parallel
    {
#if PACKED_DATA
        deleteBuffer0_(&threadLocalData0_.wyPacks);
#endif
        freeBufferImpl_(&threadLocalData0_.usedVariables);
        freeBufferImpl_(&threadLocalData0_.tree);

        freeBuffersImpl_<uint8_t>();
        freeBuffersImpl_<uint16_t>();
        freeBuffersImpl_<uint32_t>();
        freeBuffersImpl_<uint64_t>();
    }
}

template<typename T>
void TreeTrainerBase::freeBuffersImpl_()
{
    freeBufferImpl_(&threadLocalData1_<T>.orderedSamplesByVariable);
    freeBufferImpl_(&threadLocalData1_<T>.sampleBuffer);
    freeBufferImpl_(&threadLocalData1_<T>.samplePointerBuffer);
    freeBufferImpl_(&threadLocalData1_<T>.treeNodeTrainers);

    freeBufferImpl_(&threadLocalData2_<T>.sampleStatus);
}

template<typename T>
void TreeTrainerBase::freeBufferImpl_(vector<T>* t)
{
    t->clear();
    t->shrink_to_fit();
}

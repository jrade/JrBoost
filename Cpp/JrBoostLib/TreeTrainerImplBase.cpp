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
#if PACKED_DATA
        n += bufferSize0_(threadLocalData0_.wyPacks);
#endif
        n += bufferSize0_(threadLocalData0_.usedVariables);
        n += bufferSize0_(threadLocalData0_.tree);

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

    n += bufferSize0_(threadLocalData1_<T>.orderedSamplesByVariable);
    n += bufferSize0_(threadLocalData1_<T>.sampleBuffer);
    n += bufferSize0_(threadLocalData1_<T>.samplePointerBuffer);
    n += bufferSize0_(threadLocalData1_<T>.treeNodeTrainers);

    n += bufferSize0_(threadLocalData2_<T>.sampleStatus);

    return n;
}

template<typename T>
size_t TreeTrainerImplBase::bufferSize0_(const vector<T>& t)
{
    return t.capacity() * sizeof(T);
}

template<typename T>
size_t TreeTrainerImplBase::bufferSize0_(const vector<vector<T>>& t)
{
    size_t n = t.capacity() * sizeof(vector<T>);
    for (const auto& u : t)
        n += u.capacity() * sizeof(T);
    return n;
}

//......................................................................................................................

void TreeTrainerImplBase::freeBuffers()
{
#pragma omp parallel
    {
#if PACKED_DATA
        deleteBuffer0_(&threadLocalData0_.wyPacks);
#endif
        deleteBuffer0_(&threadLocalData0_.usedVariables);
        deleteBuffer0_(&threadLocalData0_.tree);

        deleteBuffersImpl_<uint8_t>();
        deleteBuffersImpl_<uint16_t>();
        deleteBuffersImpl_<uint32_t>();
        deleteBuffersImpl_<uint64_t>();
    }
}

template<typename T>
void TreeTrainerImplBase::deleteBuffersImpl_()
{
    deleteBuffer0_(&threadLocalData1_<T>.orderedSamplesByVariable);
    deleteBuffer0_(&threadLocalData1_<T>.sampleBuffer);
    deleteBuffer0_(&threadLocalData1_<T>.samplePointerBuffer);
    deleteBuffer0_(&threadLocalData1_<T>.treeNodeTrainers);

    deleteBuffer0_(&threadLocalData2_<T>.sampleStatus);
}

template<typename T>
void TreeTrainerImplBase::deleteBuffer0_(vector<T>* t)
{
    t->clear();
    t->shrink_to_fit();
}

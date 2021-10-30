//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "TreeNodeTrainer.h"


class BasePredictor;
class TreeOptions;


struct alignas(2 * sizeof(double)) WyPack
{
    double w;
    double wy;
};


class TreeTrainerImplBase
{
public:
    virtual ~TreeTrainerImplBase() = default;
    virtual unique_ptr<BasePredictor> train(CRefXd outData, CRefXd weights, const TreeOptions& options, size_t threadCount) const = 0;

    static size_t bufferSize();
    static void freeBuffers();

protected:
    TreeTrainerImplBase() = default;

// deleted:
    TreeTrainerImplBase(const TreeTrainerImplBase&) = delete;
    TreeTrainerImplBase& operator=(const TreeTrainerImplBase&) = delete;

protected:
    struct ThreadLocalData0_
    {
        ThreadLocalData0_* parent = nullptr;     // thread local data of parent thread
#if PACKED_DATA
        vector<WyPack> wyPacks;
#endif
        vector<size_t> usedVariables;
        vector<vector<TreeNodeExt>> tree;
    };

    template<typename SampleIndex>
    struct ThreadLocalData1_
    {
        ThreadLocalData1_* parent = nullptr;     // thread local data of parent thread

        vector<vector<SampleIndex>> orderedSamplesByVariable;
        // orderedSamplesByVariable[j] contains the active samples grouped by node
        // and then sorted by the j-th used variable
        // (only used if options.saveMemory() = false)

        vector<SampleIndex> sampleBuffer;

        vector<SampleIndex*> samplePointerBuffer;       // tmp buffers

        vector<CacheLineAligned<TreeNodeTrainer<SampleIndex>>> treeNodeTrainers;
    };

    template<typename SampleStatus>
    struct ThreadLocalData2_
    {
        ThreadLocalData2_* parent = nullptr;    // thread local data of parent thread

        vector<SampleStatus> sampleStatus;
        // status of each sample in the current layer of the tree
        // status`= 0 means the sample is unused
        // status = k + 1 means the sample belongs to node number k in the layer; k = 0, 1, ..., node count - 1
    };

protected:
    inline static thread_local ThreadLocalData0_ threadLocalData0_;
    template<typename SampleIndex> inline static thread_local ThreadLocalData1_<SampleIndex> threadLocalData1_;
    template<typename SampleStatus> inline static thread_local ThreadLocalData2_<SampleStatus> threadLocalData2_;

private:
    template<typename T> static size_t bufferSizeImpl_();
    template<typename T> static size_t bufferSize0_(const vector<T>& t);
    template<typename T> static size_t bufferSize0_(const vector<vector<T>>& t);
    template<typename T> static void deleteBuffersImpl_();
    template<typename T> static void deleteBuffer0_(vector<T>* t);
};

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImpl.h"

#include "ExceptionSafeOmp.h"
#include "InterruptHandler.h"
#include "pdqsort.h"
#include "StumpPredictor.h"
#include "TreeOptions.h"
#include "TreePredictor.h"
#include "TrivialPredictor.h"

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
TreeTrainerImpl<SampleIndex>::TreeTrainerImpl(CRefXXf inData, CRefXs strata) :
    inData_{ inData },
    sampleCount_{ static_cast<size_t>(inData.rows()) },
    variableCount_{ static_cast<size_t>(inData.cols()) },
    sortedSamples_{ initSortedSamples_() },
    strata_{ strata },
    stratum0Count_{ (strata == 0).cast<size_t>().sum() },
    stratum1Count_{ (strata == 1).cast<size_t>().sum() }
{
}


template<typename SampleIndex>
vector<vector<SampleIndex>> TreeTrainerImpl<SampleIndex>::initSortedSamples_() const
{
    vector<vector<SampleIndex>> sortedSamples(variableCount_);

    const size_t threadCount = std::min<size_t>(omp_get_max_threads(), variableCount_);

#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        ASSERT(static_cast<size_t>(omp_get_num_threads()) == threadCount);

        vector<pair<float, SampleIndex>> tmp{ sampleCount_ };

        const size_t threadId = omp_get_thread_num();
        const size_t jStart = variableCount_ * threadId / threadCount;
        const size_t jStop = variableCount_ * (threadId + 1) / threadCount;

        for (size_t j = jStart; j != jStop; ++j) {

            const float* pInDataColJ = std::data(inData_.col(j));
            for (size_t i = 0; i != sampleCount_; ++i)
                tmp[i] = { pInDataColJ[i], static_cast<SampleIndex>(i) };
            pdqsort_branchless(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return x.first < y.first; });
            sortedSamples[j].resize(sampleCount_);
            SampleIndex* sortedSamplesJ = data(sortedSamples[j]);
            for (size_t i = 0; i != sampleCount_; ++i)
                sortedSamplesJ[i] = tmp[i].second;
        }
    }

    return sortedSamples;
}

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
unique_ptr<BasePredictor> TreeTrainerImpl<SampleIndex>::train(
    CRefXd outData, CRefXd weights, const TreeOptions& options, size_t threadCount) const
{
    if (currentInterruptHandler != nullptr)
        currentInterruptHandler->check();

    if (abortThreads)
        throw ThreadAborted();

    // profiling zero calibration
    PROFILE::PUSH(PROFILE::ZERO);
    size_t ITEM_COUNT = 1;

    // validate data
    //PROFILE::SWITCH(ITEM_COUNT, PROFILE::VALIDATE);
    //validateData_(outData, weights);
    //ITEM_COUNT = sampleCount_;

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_USED_VARIABLES);
    size_t usedVariableCount;
    std::tie(usedVariableCount, ITEM_COUNT) = initUsedVariables_(options);

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SAMPLE_STATUS);
    ITEM_COUNT = sampleCount_;
    size_t usedSampleCount = initSampleStatus_(weights, options);

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_TREE);
    ITEM_COUNT = sampleCount_;
    initRoot_(outData, weights, usedSampleCount);

    if (threadCount == 0)
        threadCount = ::theParallelTree ? omp_get_max_threads() : 1;
    threadCount = std::min<size_t>(threadCount, omp_get_max_threads());
    threadCount = std::min<size_t>(threadCount, usedVariableCount);
    size_t threadShift = std::uniform_int_distribution<size_t>(0, threadCount - 1)(theRne);

    for (size_t d = 0; d != options.maxDepth(); ++d) {

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SPLITS);
        ITEM_COUNT = 0;
        initSplits_(options, d, threadCount);

        OuterThreadData_* out = &out_;

#pragma omp parallel num_threads(static_cast<int>(threadCount)) firstprivate(ITEM_COUNT)
        {
            ASSERT(static_cast<size_t>(omp_get_num_threads()) == threadCount);
            size_t threadIndex = omp_get_thread_num();

            in_.out = out;      // give the inner threads access to the thread local data of the outer threads

            const size_t i = (threadIndex + threadShift) % threadCount;
            // since the profiling only tracks the master thread, we randomize the iteration assigned to each thread
            const size_t usedVariableIndexStart = usedVariableCount * i / threadCount;
            const size_t usedVariableIndexStop = usedVariableCount * (i + 1) / threadCount;

            for (
                size_t usedVariableIndex = usedVariableIndexStart;
                usedVariableIndex != usedVariableIndexStop;
                ++usedVariableIndex
            ) {
                const SampleIndex* orderedSamples;
                if (d == 0) {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES_FAST);
                    ITEM_COUNT = sampleCount_;
                    orderedSamples = initOrderedSamplesFast_(usedVariableIndex, usedSampleCount, options, d);
                }
                else if (options.saveMemory()) {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES);
                    ITEM_COUNT = sampleCount_;
                    orderedSamples = initOrderedSamples_(usedVariableIndex, usedSampleCount, options, d);
                }
                else {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_ORDERED_SAMPLES);
                    ITEM_COUNT = usedSampleCount;
                    orderedSamples = updateOrderedSamples_(usedVariableIndex, usedSampleCount, options, d);
                }

                PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SPLITS);
                ITEM_COUNT = usedSampleCount;
                updateSplits_(outData, weights, options, orderedSamples, usedVariableIndex, d, threadIndex);
            }

            in_.out = nullptr;

            PROFILE::SWITCH(ITEM_COUNT, PROFILE::INNER_THREAD_SYNCH);
        }   // omp parallel
        ITEM_COUNT = 0;

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::FINALIZE_SPLITS);
        ITEM_COUNT = 0;
        joinSplits_(d, threadCount);
        size_t newNodeCount = finalizeSplits_(d);
        if (newNodeCount == 0) break;

        if (d + 1 == options.maxDepth()) break;
        // if this test is removed, then y in the child nodes is recalculated
        // with more precision in updateSampleStatus_() 

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SAMPLE_STATUS);
        ITEM_COUNT = sampleCount_;
        usedSampleCount = updateSampleStatus_(outData, weights, d);
    }   // d loop

    // TO DO: PRUNE TREE

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::CREATE_PREDICTOR);
    ITEM_COUNT = 0;
    unique_ptr<BasePredictor> predictor = initPredictor_();

    PROFILE::POP(ITEM_COUNT);

    return predictor;
};

//......................................................................................................................

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::validateData_(CRefXd outData, CRefXd weights) const
{
    if (static_cast<size_t>(outData.rows()) != sampleCount_)
        throw std::invalid_argument("Train indata and outdata have different numbers of samples.");
    if (!(outData.abs() < numeric_limits<double>::infinity()).all())
        throw std::invalid_argument("Train outdata has values that are infinity or NaN.");

    if (static_cast<size_t>(weights.rows()) != sampleCount_)
        throw std::invalid_argument("Train indata and weights have different numbers of samples.");
    if (!(weights.abs() < numeric_limits<float>::infinity()).all())
        throw std::invalid_argument("Train weights have values that are infinity or NaN.");
    if (!(weights >= 0.0).all())
        throw std::invalid_argument("Train weights have non-negative values.");
}


template<typename SampleIndex>
pair<size_t, size_t> TreeTrainerImpl<SampleIndex>::initUsedVariables_(const TreeOptions& options) const
{
    OuterThreadData_* out = &out_;
    RandomNumberEngine& rne = theRne;
    size_t candidateVariableCount = std::min(variableCount_, options.topVariableCount());
    size_t usedVariableCount = static_cast<size_t>(options.usedVariableRatio() * candidateVariableCount + 0.5);
    if (usedVariableCount == 0) usedVariableCount = 1;

    ASSERT(0 < usedVariableCount);
    ASSERT(usedVariableCount <= candidateVariableCount);
    ASSERT(candidateVariableCount <= variableCount_);

    out->usedVariables.resize(usedVariableCount);
    auto p = begin(out->usedVariables);

    size_t i = 0;
    size_t n = candidateVariableCount;
    size_t k = usedVariableCount;
    while (k > 0) {
        *p = i;
        bool b = BernoulliDistribution(k, n)(rne);
        p += b;
        k -= b;
        --n;
        ++i;
    }

    if (!options.saveMemory())
        out->sampleBufferByVariable.resize(std::max(usedVariableCount, size(out->sampleBufferByVariable)));

    return std::make_pair(usedVariableCount, candidateVariableCount);
}


template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::initSampleStatus_(CRefXd weights0, const TreeOptions& options) const
{
    size_t usedSampleCount;

    RandomNumberEngine& rne = theRne;
    OuterThreadData_* out = &out_;
    InnerThreadData_* in = &in_;

    out->sampleStatus.resize(sampleCount_);

    double minSampleWeight = options.minAbsSampleWeight();
    const double minRelSampleWeight = options.minRelSampleWeight();
    if (minRelSampleWeight > 0.0)
        minSampleWeight = std::max(minSampleWeight, weights0.maxCoeff() * minRelSampleWeight);

    if (minSampleWeight == 0.0) {

        if (options.usedSampleRatio() == 1.0) {
            usedSampleCount = sampleCount_;
            std::fill(begin(out->sampleStatus), end(out->sampleStatus), static_cast<SampleIndex>(1));
        }

        else if (!options.isStratified()) {

            // create a random mask of length n with k ones and n - k zeros
            // n = total number of samples
            // k = number of used samples

            size_t n = sampleCount_;
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0) k = 1;
            usedSampleCount = k;

            auto pBegin = begin(out->sampleStatus);
            auto pEnd = end(out->sampleStatus);
            for (auto p = pBegin; p != pEnd; ++p) {
                bool b = BernoulliDistribution(k, n) (rne);
                *p = b;
                k -= b;
                --n;
            }
        }

        else {
            // create a random mask of length n = n[0] + n[1] with k = k[0] + k[1] ones and n - k zeros
            // n[s] = total number of samples in stratum s
            // k[s] = number of used samples in stratum s

            array<size_t, 2> n{ stratum0Count_, stratum1Count_ };

            array<size_t, 2> k;
            k[0] = static_cast<size_t>(options.usedSampleRatio() * n[0] + 0.5);
            if (k[0] == 0 && n[0] > 0) k[0] = 1;
            k[1] = static_cast<size_t>(options.usedSampleRatio() * n[1] + 0.5);
            if (k[1] == 0 && n[1] > 0) k[1] = 1;
            usedSampleCount = k[0] + k[1];

            const size_t* q = std::data(strata_);
            auto pBegin = begin(out->sampleStatus);
            auto pEnd = end(out->sampleStatus);
            for (auto p = pBegin; p != pEnd; ++p, ++q) {
                size_t stratum = *q;
                bool b = BernoulliDistribution(k[stratum], n[stratum]) (rne);
                *p = b;
                k[stratum] -= b;
                --n[stratum];
            }
        }
    }

    else {  // minSampleWeight > 0.0

        // same as above, but first we identify the smaples with weight >= sampleMinWeight
        // these samples are stored in tmpSamples
        // then we select among those samples

        const double* weights = std::data(weights0);

        SampleIndex* sampleStatus = data(out->sampleStatus);

        if (options.usedSampleRatio() == 1.0) {
            usedSampleCount = 0;
            for (size_t i = 0; i != sampleCount_; ++i) {
                bool b = weights[i] >= minSampleWeight;
                sampleStatus[i] = b;
                usedSampleCount += b;
            }
        }

        else if (!options.isStratified()) {
            in->sampleBuffer.resize(sampleCount_);
            auto p = begin(in->sampleBuffer);
            for (size_t i = 0; i != sampleCount_; ++i) {
                bool b = weights[i] >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
            }
            in->sampleBuffer.resize(p - begin(in->sampleBuffer));

            size_t n = size(in->sampleBuffer);
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0 && n > 0) k = 1;
            usedSampleCount = k;

            std::fill(begin(out->sampleStatus), end(out->sampleStatus), static_cast<SampleIndex>(0));
            for (SampleIndex i : in->sampleBuffer) {
                bool b = BernoulliDistribution(k, n) (rne);
                sampleStatus[i] = b;
                k -= b;
                --n;
            }
        }

        else {
            const size_t* strata = std::data(strata_);
            array<size_t, 2> n = { 0, 0 };
            in->sampleBuffer.resize(sampleCount_);
            auto p = begin(in->sampleBuffer);
            for (size_t i = 0; i != sampleCount_; ++i) {
                size_t stratum = strata[i];
                bool b = weights[i] >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
                n[stratum] += b;
            }
            in->sampleBuffer.resize(p - begin(in->sampleBuffer));

            array<size_t, 2> k;
            k[0] = static_cast<size_t>(options.usedSampleRatio() * n[0] + 0.5);
            if (k[0] == 0 && n[0] > 0) k[0] = 1;
            k[1] = static_cast<size_t>(options.usedSampleRatio() * n[1] + 0.5);
            if (k[1] == 0 && n[1] > 0) k[1] = 1;
            usedSampleCount = k[0] + k[1];

            std::fill(begin(out->sampleStatus), end(out->sampleStatus), static_cast<SampleIndex>(0));
            for (size_t i : in->sampleBuffer) {
                size_t stratum = strata[i];
                bool b = BernoulliDistribution(k[stratum], n[stratum]) (rne);
                sampleStatus[i] = b;
                k[stratum] -= b;
                --n[stratum];
            }
        }
    }

    return usedSampleCount;
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initRoot_(
    CRefXd outData, CRefXd weights, size_t usedSampleCount) const
{
    OuterThreadData_* out = &out_;

    double sumW = 0.0;
    double sumWY = 0.0;
    const SampleIndex* sampleStatus = data(out->sampleStatus);
    const double* pOutData = std::data(outData);
    const double* pWeights = std::data(weights);
    for (size_t i = 0; i != sampleCount_; ++i) {
        double m = static_cast<double>(sampleStatus[i]);
        double w = pWeights[i];
        double y = pOutData[i];
        sumW += m * w;
        sumWY += m * w * y;
    }

    out->tree.resize(std::max<size_t>(1, size(out->tree)));
    out->tree.front().resize(1);
    TreeNodeExt* root = data(out->tree.front());
    root->isLeaf = true;
    root->sampleCount = usedSampleCount;
    root->sumW = sumW;
    root->sumWY = sumWY;
    root->y = (sumW == 0.0) ? 0.0f : static_cast<float>(sumWY / sumW);
}

//......................................................................................................................

template<typename SampleIndex>
const SampleIndex* TreeTrainerImpl<SampleIndex>::initOrderedSamplesFast_(
    size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& options, size_t d) const
{
    InnerThreadData_* in = &in_;
    OuterThreadData_* out = in->out;

    vector<TreeNodeExt>& nodes = out->tree[d];
    const size_t nodeCount = size(nodes);
    ASSERT(nodeCount == 1);

    size_t variableIndex = out->usedVariables[usedVariableIndex];

    in->sampleBuffer.resize(usedSampleCount);

    // branchfree code for copying all i in sortedSamples_[variableIndex] with sampleStatus[i] = 1 to sampleBuffer

    const SampleIndex* p = data(sortedSamples_[variableIndex]);
    SampleIndex* q = data(in->sampleBuffer);
    SampleIndex* qEnd = data(in->sampleBuffer) + size(in->sampleBuffer);
    const SampleIndex* sampleStatus = data(out->sampleStatus);
    while (q != qEnd) {
        SampleIndex i = *p;
        *q = i;
        ++p;
        q += sampleStatus[i];
    }
    q = data(in->sampleBuffer);

    if(d + 1 != options.maxDepth() && !options.saveMemory())
        swap(in->sampleBuffer, out->sampleBufferByVariable[usedVariableIndex]);

    return q;
}


template<typename SampleIndex>
const SampleIndex* TreeTrainerImpl<SampleIndex>::initOrderedSamples_(
    size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& options, size_t d) const
{
    ASSERT(options.saveMemory());

    InnerThreadData_* in = &in_;
    OuterThreadData_* out = in->out;

    const vector<TreeNodeExt>& nodes{ out->tree[d] };
    const size_t nodeCount = size(nodes);

    const size_t variableIndex = out->usedVariables[usedVariableIndex];

    in->sampleBuffer.resize(sampleCount_);

    in->samplePointerBuffer.resize(nodeCount + 1);
    SampleIndex** samplePointerBuffer = data(in->samplePointerBuffer);

    SampleIndex* p = data(in->sampleBuffer);
    const size_t unusedSampleCount = sampleCount_ - usedSampleCount;
    for (size_t k = 0; k != nodeCount + 1; ++k) {
        samplePointerBuffer[k] = p;
        p += (k == 0) ? unusedSampleCount : nodes[k - 1].sampleCount;
    }

    // samplePointerBuffer[i] (i = 0, 1, 2, ... nodeCount) points to the block
    // where we soon will store samples with status i 
    // (status i = 0 means unused sample, i > 0 means sample nbelongs to node number i - 1)

    const SampleIndex* sampleStatus = data(out->sampleStatus);
    for (SampleIndex i : sortedSamples_[variableIndex]) {
        SampleIndex s = sampleStatus[i];
        *(samplePointerBuffer[s]++) = i;
    }

    // samplePointerBuffer[i] (i = 0, 1, 2, ... nodeCount) now points to end of the above block
    // which has been filled with the appropriate samples
    // This means that samplePointerBuffer[i] (i = 0, 1, 2, ... nodeCount - 1) points to the beginning of the block
    // where we now store samples with status i + 1, i.e. that belong to node i
    // and orderedSamplesByNode[nodeCount] points to the end of the last block.

    return samplePointerBuffer[0];
}


template<typename SampleIndex>
const SampleIndex* TreeTrainerImpl<SampleIndex>::updateOrderedSamples_(
    size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& options, size_t d) const
{
    ASSERT(!options.saveMemory());

    InnerThreadData_* in = &in_;
    OuterThreadData_* out = in->out;

    vector<TreeNodeExt>& parentNodes{ out->tree[d - 1] };
    vector<TreeNodeExt>& childNodes{ out->tree[d] };
    size_t childNodeCount = size(childNodes);

    in->sampleBuffer.resize(usedSampleCount);

    in->samplePointerBuffer.resize(childNodeCount + 1);
    SampleIndex** samplePointerBuffer = data(in->samplePointerBuffer);
    SampleIndex* p = data(in->sampleBuffer);
    for (size_t k = 0; k != childNodeCount; ++k) {
        samplePointerBuffer[k + 1] = p;
        p += childNodes[k].sampleCount;
    }

    // samplePointerBuffer[i] (i = 1, 2, ... nodeCount-1) points to the block
    // where we soon will store samples with status i 
    // (status i = 0 means unused sample, i > 0 means sample nbelongs to node number i - 1)

    p = data(out->sampleBufferByVariable[usedVariableIndex]);
    const SampleIndex* sampleStatus = data(out->sampleStatus);
    for (const TreeNodeExt& parentNode : parentNodes) {
        SampleIndex* pEnd = p + parentNode.sampleCount;
        if (parentNode.isLeaf)
            p = pEnd;
        else
            for (; p != pEnd; ++p) {
                SampleIndex i = *p;
                SampleIndex s = sampleStatus[i];
                *(samplePointerBuffer[s]++) = i;
            }
    }
    p = data(in->sampleBuffer);

    if (d + 1 != options.maxDepth())
        swap(out->sampleBufferByVariable[usedVariableIndex], in->sampleBuffer);

    return p;
}

//......................................................................................................................

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initSplits_(const TreeOptions& options, size_t d, size_t threadCount) const
{
    OuterThreadData_* out = &out_;

    const vector<TreeNodeExt>& parentNodes{ out->tree[d] };
    const size_t parentNodeCount = size(parentNodes);

    out->treeNodeTrainers.resize(threadCount * (parentNodeCount + 1));
    // parentNodeCount elements plus one dummy element to avoid sharing of cache lines between threads

    size_t k = 0;
    for (size_t i = 0; i != threadCount; ++i, ++k)
        for (size_t j = 0; j != parentNodeCount; ++j, ++k)
            out->treeNodeTrainers[k].init(&parentNodes[j], options);
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateSplits_(
    CRefXd outData,
    CRefXd weights,
    const TreeOptions& options,
    const SampleIndex* orderedSamples,
    size_t usedVariableIndex,
    size_t d,
    size_t threadIndex
) const
{
    InnerThreadData_* in = &in_;
    OuterThreadData_* out = in->out;

    const vector<TreeNodeExt>& parentNodes{ out->tree[d] };
    size_t parentNodeCount = size(parentNodes);

    size_t variableIndex = out->usedVariables[usedVariableIndex];

    const SampleIndex* p0 = orderedSamples;
    size_t k = threadIndex * (parentNodeCount + 1);
    for (size_t j = 0; j != parentNodeCount; ++j, ++k) {
        const SampleIndex * p1 = p0 + parentNodes[j].sampleCount;
        out->treeNodeTrainers[k].update(
            inData_, outData, weights, options, p0, p1, variableIndex);
        p0 = p1;
    }
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::joinSplits_(size_t d, size_t threadCount) const
{
    OuterThreadData_* out = &out_;

    const vector<TreeNodeExt>& parentNodes{ out->tree[d] };
    const size_t parentNodeCount = size(parentNodes);

    size_t k = parentNodeCount + 1;
    for (size_t i = 1; i != threadCount; ++i, ++k)
        for (size_t j = 0; j != parentNodeCount; ++j, ++k)
            out->treeNodeTrainers[j].join(out->treeNodeTrainers[k]);
}


template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::finalizeSplits_(size_t d) const
{
    OuterThreadData_* out = &out_;

    out->tree.resize(std::max(d + 2, size(out->tree)));
    vector<TreeNodeExt>& parentNodes{ out->tree[d] };
    vector<TreeNodeExt>& childNodes{ out->tree[d + 1] };

    size_t parentNodeCount = size(parentNodes);
    size_t maxChildNodeCount = 2 * parentNodeCount;
    childNodes.resize(maxChildNodeCount);

    TreeNodeExt* parentNode = data(parentNodes);
    TreeNodeExt* childNode = data(childNodes);
    for (size_t k = 0; k != parentNodeCount; ++k)
        out->treeNodeTrainers[k].finalize(&parentNode, &childNode);

    size_t childNodeCount = childNode - data(childNodes);
    childNodes.resize(childNodeCount);
    return childNodeCount;
}

//......................................................................................................................

template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::updateSampleStatus_(CRefXd outData, CRefXd weights, size_t d) const
{
    OuterThreadData_* out = &out_;

    const TreeNodeExt* parentNodes{ data(out->tree[d]) };
    vector<TreeNodeExt>& childNodes{ out->tree[d + 1] };
    SampleIndex* sampleStatus{ data(out->sampleStatus) };

    for (TreeNodeExt& childNode : childNodes) {
        childNode.sampleCount = 0;
        childNode.sumW = 0.0;
        childNode.sumWY = 0.0;
    }

    const double* pOutData = std::data(outData);
    const double* pWeights = std::data(weights);

    for (size_t i = 0; i != sampleCount_; ++i) {

        SampleIndex s = sampleStatus[i];
        if (s == 0) continue;

        const TreeNodeExt* parentNode = &parentNodes[s - 1];
        if (parentNode->isLeaf) {
            sampleStatus[i] = 0;
            continue;
        }

        TreeNodeExt* childNode = (inData_(i, parentNode->j) < parentNode->x)
            ? static_cast<TreeNodeExt*>(parentNode->leftChild)
            : static_cast<TreeNodeExt*>(parentNode->rightChild);
        s = static_cast<SampleIndex>(childNode - data(childNodes) + 1);
        sampleStatus[i] = s;

        double w = pWeights[i];
        double y = pOutData[i];
        ++childNode->sampleCount;
        childNode->sumW += w;
        childNode->sumWY += w * y;
    }

    size_t usedSampleCount = 0;
    for (TreeNodeExt& childNode : childNodes) {
        childNode.y = (childNode.sumW == 0) ? 0.0f : static_cast<float>(childNode.sumWY / childNode.sumW);
        usedSampleCount += childNode.sampleCount;
    }
    return usedSampleCount;
}


template<typename SampleIndex>
unique_ptr<BasePredictor> TreeTrainerImpl<SampleIndex>::initPredictor_() const
{
    OuterThreadData_* out = &out_;

    unique_ptr<BasePredictor> pred;
    const TreeNodeExt* root = data(out->tree.front());
    size_t treeDepth = depth(root);

    if (treeDepth == 0)
        pred = std::make_unique<TrivialPredictor>(root->y);
    else if (treeDepth == 1)
        pred = std::make_unique<StumpPredictor>(root->j, root->x, root->leftChild->y, root->rightChild->y, root->gain);
    else {
        vector<TreeNode> clonedNodes = cloneBreadthFirst(root);     // depth or breadth does not matter
        const TreeNode* clonedRoot = data(clonedNodes);
        pred = std::make_unique<TreePredictor>(clonedRoot, move(clonedNodes));
    }

    return pred;
}

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImpl<uint8_t>;
template class TreeTrainerImpl<uint16_t>;
template class TreeTrainerImpl<uint32_t>;
template class TreeTrainerImpl<uint64_t>;

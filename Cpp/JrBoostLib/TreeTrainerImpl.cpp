//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImpl.h"

#include "ExceptionSafeOmp.h"
#include "InterruptHandler.h"
#include "StumpPredictor.h"
#include "TreeOptions.h"
#include "TreePredictor.h"
#include "TrivialPredictor.h"

/*
    Overview:
        This code trains a tree predictor. The method is pretty straightforward.
        Start with layer 0 consting of the root only.
        Then iteratively split the nodes in each layer by testing all possible splits,
        maximizing a gain function subject to various constraints as described in any machine learning text book.
   
        The tricky part is keeping track of the samples.
        In each layer of the tree each sample is either unused or assigned to one of the nodes.
        When splitting a node, we need to know which samples assigned to that node and then sort them
        with respect to each of the variables used.

        Thus we define the status of a sample in layer d as
            0 if the sample is not used
            k + 1 if the sample is assigned to node number k in the layer.

        Note that for layer 0 the status of every sample is 0 or 1; this observation is used in the implementation of
        initSampleStatus_(), initRoot_(), and initOrderedSamplesLayer0_().

        In the constructor we sort all samples with respect to each variable once and for all.
        No further sorting is done in the code, we just extract sorted sublists from these presorted lists.

        The main tasks carried out by the code are:
            1. Initialize and update a vector that contains the status of each sample in the current layer of the tree.
            2. Using 1, initialize and update a buffer (or buffers) where the samples are grouped according node (or status)
                and then each group is sorted according to a variable (or all variables).
                The buffer (or buffers) may or may not contain the unused samples (status = 0) as an initial group.
            3. Using the ouput from 2, find the best split of each node in the current layer.

    One letter variable names:
        d depth (used to index the layers in the tree, d = 0 being the root)
        i sample index
        j variable index
        k node and node trainer index
        s sample status
        t threadlocal data
        w weight
        x indata value
        y outdata value
        z sample stratum

     Optimizations:
        fast sorting algorithm (not std::sort)
        fast random number generator (not std::mt19937)
        fast Bernoulli distribution (not std::bernoulli_distribution)
        branchfree code
            the only if-statements in inner loops are in updateSampleStatus_() 
            and TreeNodeTrainer::update(), and the latter is fairly predictable
        memory optimized storage of vectors of sample indices
            using the template parameter SampleIndex
        very few memory allocations
            using threadlocal buffers that expand as needed and never shrink
            besides these buffers, the only memory allocations are in the constructor and in createPredictor_()
*/

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
TreeTrainerImpl<SampleIndex>::TreeTrainerImpl(CRefXXfc inData, CRefXs strata) :
    inData_{ inData },
    sampleCount_{ static_cast<size_t>(inData.rows()) },
    variableCount_{ static_cast<size_t>(inData.cols()) },
    sortedSamples_{ createSortedSamples_() },
    strata_{ strata },
    stratum0Count_{ (strata == 0).cast<size_t>().sum() },
    stratum1Count_{ (strata == 1).cast<size_t>().sum() }
{
}

// The next function creates lists of sorted samples for each variable.
// These lists are then used by initSortedSamplesFast_() and initSortedSamples_().

template<typename SampleIndex>
vector<vector<SampleIndex>> TreeTrainerImpl<SampleIndex>::createSortedSamples_() const
{
    vector<vector<SampleIndex>> sortedSamples(variableCount_);

    const size_t threadCount = std::min<size_t>(omp_get_max_threads(), variableCount_);

#pragma omp parallel num_threads(static_cast<int>(threadCount))
    {
        ASSERT(static_cast<size_t>(omp_get_num_threads()) == threadCount);

        vector<pair<float, SampleIndex>> tmp(sampleCount_);

        const size_t threadId = omp_get_thread_num();
        const size_t jStart = variableCount_ * threadId / threadCount;
        const size_t jStop = variableCount_ * (threadId + 1) / threadCount;

        for (size_t j = jStart; j != jStop; ++j) {

            const float* pInDataColJ = std::data(inData_.col(j));
            for (size_t i = 0; i != sampleCount_; ++i)
                tmp[i] = { pInDataColJ[i], static_cast<SampleIndex>(i) };
            fastSort(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return x.first < y.first; });

            sortedSamples[j].resize(sampleCount_);
            SampleIndex* pSortedSamplesJ = data(sortedSamples[j]);
            for (size_t i = 0; i != sampleCount_; ++i)
                pSortedSamplesJ[i] = tmp[i].second;
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

    size_t usedVariableCount = usedVariableCount_(options);

    if (threadCount == 0)
        threadCount = ::globParallelTree ? omp_get_max_threads() : 1;
    threadCount = std::min<size_t>(threadCount, omp_get_max_threads());
    threadCount = std::min<size_t>(threadCount, usedVariableCount);
    size_t threadShift = std::uniform_int_distribution<size_t>(0, threadCount - 1)(theRne);

    //// validate data
    //PROFILE::PUSH(PROFILE::VALIDATE);
    //validateData_(outData, weights);
    //size_t ITEM_COUNT = sampleCount_;

    PROFILE::PUSH(PROFILE::INIT_SAMPLE_STATUS);
    size_t ITEM_COUNT = sampleCount_;
    initSampleStatus_(weights, options);

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_TREE);
    ITEM_COUNT = sampleCount_;
    size_t usedSampleCount = initRoot_(outData, weights);

    for (size_t d = 0; d != options.maxDepth(); ++d) {

        if (d == 0 || options.selectVariablesByLevel()) {
            PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_USED_VARIABLES);
            ITEM_COUNT = initUsedVariables_(options);
        }

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SPLITS);
        ITEM_COUNT = 0;
        initNodeTrainers_(options, d, threadCount);

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::PROFILE::INNER_THREAD_SYNCH);
        ITEM_COUNT = 0;

        ThreadLocalData_& t = threadLocalData_;

#pragma omp parallel num_threads(static_cast<int>(threadCount)) firstprivate(ITEM_COUNT)
        {
            ASSERT(static_cast<size_t>(omp_get_num_threads()) == threadCount);

            threadLocalData_.parent = &t;      // give the inner threads access to the thread local data of the outer thread

            const size_t threadIndex = omp_get_thread_num();
            const size_t shiftedThreadIndex = (threadIndex + threadShift) % threadCount;
            // since the profiling only tracks the master thread, we randomize the variables assigned to each thread
            const size_t usedVariableIndexStart = usedVariableCount * shiftedThreadIndex / threadCount;
            const size_t usedVariableIndexStop = usedVariableCount * (shiftedThreadIndex + 1) / threadCount;

            for (
                size_t usedVariableIndex = usedVariableIndexStart;
                usedVariableIndex != usedVariableIndexStop;
                ++usedVariableIndex
            ) {
                const SampleIndex* pOrderedSamples;

                if (d == 0) {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES_FAST);
                    ITEM_COUNT = sampleCount_;
                    pOrderedSamples = initOrderedSamplesLayer0_(usedVariableIndex, usedSampleCount, options, d);
                }
                else if (options.saveMemory() || options.selectVariablesByLevel()) {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES);
                    ITEM_COUNT = sampleCount_;
                    pOrderedSamples = initOrderedSamples_(usedVariableIndex, usedSampleCount, options, d);
                }
                else {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_ORDERED_SAMPLES);
                    ITEM_COUNT = usedSampleCount;
                    pOrderedSamples = updateOrderedSamples_(usedVariableIndex, usedSampleCount, options, d);
                }

                PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SPLITS);
                ITEM_COUNT = usedSampleCount;
                updateNodeTrainers_(outData, weights, pOrderedSamples, usedVariableIndex, d);

            }   // end variable index loop
            threadLocalData_.parent = nullptr;
            PROFILE::SWITCH(ITEM_COUNT, PROFILE::INNER_THREAD_SYNCH);
        }   // end omp parallel
        ITEM_COUNT = 0;

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::FINALIZE_SPLITS);
        ITEM_COUNT = 0;
        joinNodeTrainers_(d, threadCount);
        usedSampleCount = finalizeNodeTrainers_(d);
        if (usedSampleCount == 0) break;

        if (d + 1 == options.maxDepth()) break;
        // if this test is removed, then y in the child nodes is recalculated
        // with more precision in updateSampleStatus_() 

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SAMPLE_STATUS);
        ITEM_COUNT = sampleCount_;
        updateSampleStatus_(outData, weights, d);

    }   // end d loop

    // TO DO: PRUNE TREE

    // profiling zero calibration
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::ZERO);
    ITEM_COUNT = 1;

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::CREATE_PREDICTOR);
    ITEM_COUNT = 0;
    unique_ptr<BasePredictor> predictor = createPredictor_();

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
    if (!(weights.abs() < numeric_limits<double>::infinity()).all())
        throw std::invalid_argument("Train weights have values that are infinity or NaN.");
    if (!(weights >= 0.0).all())
        throw std::invalid_argument("Train weights have non-negative values.");
}

template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::usedVariableCount_(const TreeOptions& options) const
{
    size_t candidateVariableCount = std::min(variableCount_, options.topVariableCount());
    size_t usedVariableCount = static_cast<size_t>(std::round(options.usedVariableRatio() * candidateVariableCount));
    if (usedVariableCount == 0) usedVariableCount = 1;

    ASSERT(0 < usedVariableCount);
    ASSERT(usedVariableCount <= candidateVariableCount);
    ASSERT(candidateVariableCount <= variableCount_);

    return usedVariableCount;
}

template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::initUsedVariables_(const TreeOptions& options) const
{
    ThreadLocalData_& t = threadLocalData_;
    RandomNumberEngine& rne = theRne;
    size_t candidateVariableCount = std::min(variableCount_, options.topVariableCount());
    size_t usedVariableCount = usedVariableCount_(options);

    t.usedVariables.resize(usedVariableCount);
    size_t* pUsedVariables = data(t.usedVariables);

    size_t j = 0;
    size_t n = candidateVariableCount;
    size_t m = usedVariableCount;
    while (m != 0) {
        *pUsedVariables = j;
        bool b = BernoulliDistribution_(m, n)(rne);
        pUsedVariables += b;
        m -= b;
        --n;
        ++j;
    }

    if (!options.saveMemory() && !options.selectVariablesByLevel() && options.maxDepth() != 1)
        t.orderedSamplesByVariable.resize(std::max(usedVariableCount, size(t.orderedSamplesByVariable)));

    return candidateVariableCount;
}

// The next function creates the root of the tree.
// It also calculates sampleCount, sumW and sumWY for the root.

template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::initRoot_(
    CRefXd outData, CRefXd weights) const
{
    ThreadLocalData_& t = threadLocalData_;

    const SampleStatus* pSampleStatus = data(t.sampleStatus);
    const double* pOutData = std::data(outData);
    const double* pWeights = std::data(weights);

    size_t usedSampleCount = 0;
    double sumW = 0.0;
    double sumWY = 0.0;
    for (size_t i = 0; i != sampleCount_; ++i) {
        SampleStatus s = pSampleStatus[i];       // 0 or 1
        double w = pWeights[i];
        double y = pOutData[i];
        usedSampleCount += s;
        sumW += s * w;
        sumWY += s * w * y;
    }

    if (empty(t.tree)) {
        t.tree.resize(1);
        t.tree.front().resize(1);
    }

    TreeNodeExt& root = t.tree.front().front();
    root.isLeaf = true;
    root.sampleCount = usedSampleCount;
    root.sumW = sumW;
    root.sumWY = sumWY;
    root.y = (sumW == 0.0) ? 0.0f : static_cast<float>(sumWY / sumW);

    return usedSampleCount;
}

//......................................................................................................................

// The next function randomly assigns status 0 (unused) or 1 (belongs to the root) to each sample

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initSampleStatus_(CRefXd weights, const TreeOptions& options) const
{
    RandomNumberEngine& rne = theRne;
    ThreadLocalData_& t = threadLocalData_;

    t.sampleStatus.resize(sampleCount_);
    SampleStatus* pSampleStatus = data(t.sampleStatus);

    double minSampleWeight = options.minAbsSampleWeight();
    const double minRelSampleWeight = options.minRelSampleWeight();
    if (minRelSampleWeight > 0.0)
        minSampleWeight = std::max(minSampleWeight, weights.maxCoeff() * minRelSampleWeight);

    if (minSampleWeight == 0.0) {

        if (options.usedSampleRatio() == 1.0) {
            // select all samples
            std::fill(pSampleStatus, pSampleStatus + sampleCount_, static_cast<SampleStatus>(1));
        }

        else if (!options.isStratified()) {

            size_t n = sampleCount_;
            size_t m = static_cast<size_t>(std::round(options.usedSampleRatio() * sampleCount_));

            // randomly select m of n samples
            for (size_t i = 0; i != sampleCount_; ++i) {
                bool s = BernoulliDistribution_(m, n) (rne);
                pSampleStatus[i] = s;
                m -= s;
                --n;
            }
            ASSERT(m == 0);
            // for this assert to hold it is critical that BernoulliDistribution(0, n) with n > 0 always returns false
            // and that BernoulliDistribution(n, n) with n > 0 always returns true
        }

        else {
            const size_t* pStrata = std::data(strata_);

            // n[z] number of samples in stratum z
            array<size_t, 2> n{ stratum0Count_, stratum1Count_ };

            // m[z] number of used samples in stratum z
            array<size_t, 2> m{
                static_cast<size_t>(std::round(options.usedSampleRatio() * n[0])),
                static_cast<size_t>(std::round(options.usedSampleRatio() * n[1]))
            };

            // randomly select m[z] of n[z] samples from each stratum
            for (size_t i = 0; i != sampleCount_; ++i) {
                size_t z = pStrata[i];
                bool s = BernoulliDistribution_(m[z], n[z]) (rne);
                pSampleStatus[i] = s;
                m[z] -= s;
                --n[z];
            }
            ASSERT(m[0] == 0 && m[1] == 0);
        }
    }

    else {  // minSampleWeight > 0.0

        const double* pWeights = std::data(weights);

        if (options.usedSampleRatio() == 1.0) {
            // select all samples with weight >= min weight
            for (size_t i = 0; i != sampleCount_; ++i) {
                bool s = pWeights[i] >= minSampleWeight;
                pSampleStatus[i] = s;
            }
        }

        else if (!options.isStratified()) {

            // store all samples with weight >= min weight in sample buffer
            t.sampleBuffer.resize(sampleCount_);
            SampleIndex* p = data(t.sampleBuffer);
            for (size_t i = 0; i != sampleCount_; ++i) {
                *p = static_cast<SampleIndex>(i);
                bool b = pWeights[i] >= minSampleWeight;
                p += b;
            }
            t.sampleBuffer.resize(p - data(t.sampleBuffer));

            // n = number of samples with weight >= min weight
            size_t n = size(t.sampleBuffer);

            // m = numer of used samples
            size_t m = static_cast<size_t>(std::round(options.usedSampleRatio() * n));

            // randomly select m of n samples with weight >= min weight
            std::fill(pSampleStatus, pSampleStatus + sampleCount_, static_cast<SampleStatus>(0));
            for (SampleIndex i : t.sampleBuffer) {
                bool s = BernoulliDistribution_(m, n) (rne);
                pSampleStatus[i] = s;
                m -= s;
                --n;
            }
            ASSERT(m == 0);
        }

        else {
            const size_t* pStrata = std::data(strata_);

            // store all samples with weight >= min weight in sample buffer
            // n[z] = number of samples in stratum z with weight >= min weight
            array<size_t, 2> n = { 0, 0 };
            t.sampleBuffer.resize(sampleCount_);
            SampleIndex* p = data(t.sampleBuffer);
            for (size_t i = 0; i != sampleCount_; ++i) {
                *p = static_cast<SampleIndex>(i);
                bool b = pWeights[i] >= minSampleWeight;
                p += b;
                size_t z = pStrata[i];
                n[z] += b;
            }
            t.sampleBuffer.resize(p - data(t.sampleBuffer));

            // m[z] = number of used samples in stratum z
            array<size_t, 2> m{
                static_cast<size_t>(std::round(options.usedSampleRatio() * n[0])),
                static_cast<size_t>(std::round(options.usedSampleRatio() * n[1]))
            };

            // randomly select m[z] of n[z] samples with weight >= min weight from each stratum
            std::fill(pSampleStatus, pSampleStatus + sampleCount_, static_cast<SampleStatus>(0));
            for (SampleIndex i : t.sampleBuffer) {
                size_t z = pStrata[i];
                bool s = BernoulliDistribution_(m[z], n[z]) (rne);
                pSampleStatus[i] = s;
                m[z] -= s;
                --n[z];
            }
            ASSERT(m[0] == 0 && m[1] == 0);
        }
    }
}

// The next function updates the status of each sample based on layer d to the status based on layer d + 1.
// It also recalculates nodeCount, sumW, sumWY any y with better precision for each node in layer d + 1.

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateSampleStatus_(CRefXd outData, CRefXd weights, size_t d) const
{
#define PRECISE_SUMS 1

    ThreadLocalData_& t = threadLocalData_;

    const TreeNodeExt* pParentNodes = data(t.tree[d]);
    vector<TreeNodeExt>& childNodes = t.tree[d + 1];
    TreeNodeExt* pChildNodes = data(childNodes);
    SampleStatus* pSampleStatus = data(t.sampleStatus);

#if PRECISE_SUMS
    for (TreeNodeExt& childNode : childNodes) {
        childNode.sampleCount = 0;
        childNode.sumW = 0.0;
        childNode.sumWY = 0.0;
    }

    const double* pOutData = std::data(outData);
    const double* pWeights = std::data(weights);
#endif

    for (size_t i = 0; i != sampleCount_; ++i) {

        SampleStatus s = pSampleStatus[i];
        if (s == 0) continue;

        const TreeNodeExt* pParentNode = &pParentNodes[s - 1];
        if (pParentNode->isLeaf) {
            pSampleStatus[i] = 0;
            continue;
        }

        TreeNodeExt* pChildNode = (inData_(i, pParentNode->j) < pParentNode->x)
            ? static_cast<TreeNodeExt*>(pParentNode->leftChild)
            : static_cast<TreeNodeExt*>(pParentNode->rightChild);
        s = static_cast<SampleStatus>((pChildNode - pChildNodes) + 1);
        pSampleStatus[i] = s;

#if PRECISE_SUMS
        double w = pWeights[i];
        double y = pOutData[i];
        ++pChildNode->sampleCount;
        pChildNode->sumW += w;
        pChildNode->sumWY += w * y;
#endif
    }

#if PRECISE_SUMS
    for (TreeNodeExt& childNode : childNodes)
        childNode.y = (childNode.sumW == 0) ? 0.0f : static_cast<float>(childNode.sumWY / childNode.sumW);
#endif
}

//......................................................................................................................

// The following three functions all return a pointer to a buffer that contains all samples that are used in layer d
// of the tree. The buffer is divided into blocks. Block number k = 0, 1, ..., nodeCount-1 contains the samples that
// belong to node k in layer d of the tree. Each block is then sorted according to variable j.
// 
// This buffer is then used as input to updateNodeTrainers_().
//
// The function initOrderedSamplesLayer0_() does this for layer 0 which consists of the root only.
// The function uses sortedSamples_[j] which contains all samples sorted according to variable j.
//
// The function initOrderedSamples_() does the same thing for the general case with any number of nodes.
// It also uses sortedSamples_[j].
// Note that this function actually creates a buffer where the first block contains the unused samples,
// but it returns a pointer to the beginning of the next block.
//
// The function updateOrderedSamples_() does the same thing but uses the ordered samples for variable j and layer d - 1
// instead of sortedSamples_[j]. This is often faster but requires more memory.


template<typename SampleIndex>
const SampleIndex* TreeTrainerImpl<SampleIndex>::initOrderedSamplesLayer0_(
    size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& options, size_t d) const
{
    // called from the inner threads so be careful to distinguish between t and t.parent

    ThreadLocalData_& t = threadLocalData_;

    vector<TreeNodeExt>& nodes = t.parent->tree[d];
    const size_t nodeCount = size(nodes);
    ASSERT(nodeCount == 1);

    size_t j = t.parent->usedVariables[usedVariableIndex];

    t.sampleBuffer.resize(usedSampleCount);

    const SampleIndex* pSortedSamples = data(sortedSamples_[j]);
    SampleIndex* pOrderedSamples = data(t.sampleBuffer);
    SampleIndex* pOrderedSamplesEnd = data(t.sampleBuffer) + usedSampleCount;
    const SampleStatus* pSampleStatus = data(t.parent->sampleStatus);
    // slightly tricky branch-free code
    while (pOrderedSamples != pOrderedSamplesEnd) {
        SampleIndex i = *pSortedSamples;
        ++pSortedSamples;
        *pOrderedSamples = i;
        SampleStatus s = pSampleStatus[i];    // 0 or 1
        pOrderedSamples += s;
    }

    pOrderedSamples = data(t.sampleBuffer);
    if (d + 1 != options.maxDepth() && !options.saveMemory() && !options.selectVariablesByLevel())
        // save the ordered samples to be used as input when ordering samples for the next layer
        swap(t.sampleBuffer, t.parent->orderedSamplesByVariable[usedVariableIndex]);
    return pOrderedSamples;
}


template<typename SampleIndex>
const SampleIndex* TreeTrainerImpl<SampleIndex>::initOrderedSamples_(
    size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& /*options*/, size_t d) const
{
    // called from the inner threads so be careful to distinguish between t and t.parent

    ThreadLocalData_& t = threadLocalData_;

    const vector<TreeNodeExt>& nodes = t.parent->tree[d];
    const size_t nodeCount = size(nodes);
    const size_t unusedSampleCount = sampleCount_ - usedSampleCount;
    const size_t j = t.parent->usedVariables[usedVariableIndex];
    const SampleStatus* pSampleStatus = data(t.parent->sampleStatus);

    t.sampleBuffer.resize(sampleCount_);
    SampleIndex* pOrderedSamples = data(t.sampleBuffer);

    t.samplePointerBuffer.resize(nodeCount + 1);
    SampleIndex** ppOrderedSamples = data(t.samplePointerBuffer);

    for (size_t k = 0; k != nodeCount + 1; ++k) {
        ppOrderedSamples[k] = pOrderedSamples;
        pOrderedSamples += (k == 0) ? unusedSampleCount : nodes[k - 1].sampleCount;
        // add the number of samples with status = k
    }

    // samplePointerBuffer[k] (k = 0, 1, 2, ..., nodeCount) points to the beginning of the block
    // where we will store samples with status = k sorted according to variable j

    for (SampleIndex i : sortedSamples_[j]) {
        SampleStatus s = pSampleStatus[i];
        *(ppOrderedSamples[s]++) = i;
    }

    pOrderedSamples = data(t.sampleBuffer) + unusedSampleCount;
    return pOrderedSamples;
}


template<typename SampleIndex>
const SampleIndex* TreeTrainerImpl<SampleIndex>::updateOrderedSamples_(
    size_t usedVariableIndex, size_t usedSampleCount, const TreeOptions& options, size_t d) const
{
    // called from inner threads; be careful to distinguish between t and t.parent

    ThreadLocalData_& t = threadLocalData_;

    const vector<TreeNodeExt>& prevNodes = t.parent->tree[d - 1];
    const vector<TreeNodeExt>& nodes = t.parent->tree[d];
    const size_t nodeCount = size(nodes);
    const SampleStatus* pSampleStatus = data(t.parent->sampleStatus);
    const SampleIndex* pPrevOrderedSamples = data(t.parent->orderedSamplesByVariable[usedVariableIndex]);

    t.sampleBuffer.resize(usedSampleCount);
    SampleIndex* pOrderedSamples = data(t.sampleBuffer);

    t.samplePointerBuffer.resize(nodeCount + 1);
    SampleIndex** ppOrderedSamples = data(t.samplePointerBuffer);

    ppOrderedSamples[0] = nullptr;
    for (size_t k = 0; k != nodeCount; ++k) {
        ppOrderedSamples[k + 1] = pOrderedSamples;
        pOrderedSamples += nodes[k].sampleCount;
    }

    // samplePointerBuffer[k] (k = 1, 2, ..., nodeCount - 1) points to the beginning of the block
    // where we will store samples with status = k sorted according to variable j
    // samplePointerBuffer[0] is not used, we simply skip the unused samples

    for (const TreeNodeExt& prevNode : prevNodes) {
        const SampleIndex* pBlockEnd = pPrevOrderedSamples + prevNode.sampleCount;
        if (prevNode.isLeaf)
            // skip these samples, they are not used in layer d
            pPrevOrderedSamples = pBlockEnd;
        else
            // process these samples, they are used in layer d
            for (; pPrevOrderedSamples != pBlockEnd; ++pPrevOrderedSamples) {
                SampleIndex i = *pPrevOrderedSamples;
                SampleStatus s = pSampleStatus[i];
                *(ppOrderedSamples[s]++) = i;
            }
    }

    pOrderedSamples = data(t.sampleBuffer);
    if (d + 1 != options.maxDepth())
        // save the ordered samples to be used as input when ordering samples for the next layer
        swap(t.parent->orderedSamplesByVariable[usedVariableIndex], t.sampleBuffer);
    return pOrderedSamples;
}

//......................................................................................................................

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initNodeTrainers_(const TreeOptions& options, size_t d, size_t threadCount) const
{
    ThreadLocalData_& t = threadLocalData_;

    const vector<TreeNodeExt>& parentNodes = t.tree[d];
    const size_t parentNodeCount = size(parentNodes);
    t.treeNodeTrainers.resize(threadCount * parentNodeCount);

    for (size_t k = 0; k != parentNodeCount; ++k)
        t.treeNodeTrainers[k].init(parentNodes[k], options);
    for (size_t threadIndex = 1; threadIndex != threadCount; ++threadIndex) {
        const size_t k0 = threadIndex * parentNodeCount;
        for (size_t k = 0; k != parentNodeCount; ++k)
            t.treeNodeTrainers[k0 + k].init(t.treeNodeTrainers[k]);
    }
}

// The next function determines the best split of each node in layer d with respect to variable j.

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateNodeTrainers_(
    CRefXd outData,
    CRefXd weights,
    const SampleIndex* orderedSamples,
    size_t usedVariableIndex,
    size_t d
) const
{
    // called from inner threads; be careful to distinguish between t and t.parent

    ThreadLocalData_& t = threadLocalData_;

    const vector<TreeNodeExt>& parentNodes = t.parent->tree[d];
    const size_t parentNodeCount = size(parentNodes);

    const size_t threadIndex = omp_get_thread_num();
    const size_t k0 = threadIndex * parentNodeCount;
    const SampleIndex* p0 = orderedSamples;
    size_t j = t.parent->usedVariables[usedVariableIndex];

    for (size_t k = 0; k != parentNodeCount; ++k) {
        const SampleIndex * p1 = p0 + parentNodes[k].sampleCount;
        t.parent->treeNodeTrainers[k0 + k].update(
            inData_, outData, weights, p0, p1, j);
        p0 = p1;
    }
}

// The next function combines the best split found by each thread to a single best split for each node.

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::joinNodeTrainers_(size_t d, size_t threadCount) const
{
    ThreadLocalData_& t = threadLocalData_;

    const vector<TreeNodeExt>& parentNodes = t.tree[d];
    const size_t parentNodeCount = size(parentNodes);

    for (size_t threadIndex = 1; threadIndex != threadCount; ++threadIndex) {
        const size_t k0 = threadIndex * parentNodeCount;
        for (size_t k = 0; k != parentNodeCount; ++k)
            t.treeNodeTrainers[k].join(t.treeNodeTrainers[k0 + k]);
    }
}

// The next function creates layer d + 1 in the tree based on the best split for each node in layer d
// It also calculates y for each node in layer d + 1.
// It returns the number of samples used by the nodes in the next layer.

template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::finalizeNodeTrainers_(size_t d) const
{
    ThreadLocalData_& t = threadLocalData_;

    size_t usedSampleCount = 0;

    t.tree.resize(std::max(d + 2, size(t.tree)));
    vector<TreeNodeExt>& parentNodes = t.tree[d];
    vector<TreeNodeExt>& childNodes = t.tree[d + 1];

    const size_t parentNodeCount = size(parentNodes);
    const size_t maxChildNodeCount = 2 * parentNodeCount;
    childNodes.resize(maxChildNodeCount);

    TreeNodeExt* pParentNode = data(parentNodes);
    TreeNodeExt* pChildNode = data(childNodes);
    for (size_t k = 0; k != parentNodeCount; ++k)
        usedSampleCount += t.treeNodeTrainers[k].finalize(&pParentNode, &pChildNode);

    const size_t childNodeCount = pChildNode - data(childNodes);
    childNodes.resize(childNodeCount);

    return usedSampleCount;
}

//......................................................................................................................

template<typename SampleIndex>
unique_ptr<BasePredictor> TreeTrainerImpl<SampleIndex>::createPredictor_() const
{
    ThreadLocalData_& t = threadLocalData_;

    unique_ptr<BasePredictor> pred;
    const TreeNodeExt& root = t.tree.front().front();
    size_t treeDepth = depth(&root);

    if (treeDepth == 0)
        // one memory allocation
        pred = std::make_unique<TrivialPredictor>(root.y);
    else if (treeDepth == 1)
        // one memory allocation
        pred = std::make_unique<StumpPredictor>(root.j, root.x, root.leftChild->y, root.rightChild->y, root.gain);
    else {
        // two memory allocations
        vector<TreeNode> clonedNodes = cloneDepthFirst(&root);
        const TreeNode& clonedRoot = clonedNodes.front();
        pred = std::make_unique<TreePredictor>(&clonedRoot, move(clonedNodes));
    }

    return pred;
}

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImpl<uint8_t>;
template class TreeTrainerImpl<uint16_t>;
template class TreeTrainerImpl<uint32_t>;
template class TreeTrainerImpl<uint64_t>;

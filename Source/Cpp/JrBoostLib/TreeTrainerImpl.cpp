//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "TreeTrainerImpl.h"

#include "BaseOptions.h"
#include "BasePredictor.h"
#include "OmpParallel.h"


/*
Overview:
    This code trains a tree predictor. The method is pretty straightforward.
    Start with layer 0 consting of the root only.
    Then iteratively split the nodes in each layer by testing all possible splits,
    maximizing a gain function subject to various constraints as described in any machine learning text book.

    The tricky part is keeping track of the samples.
    In each layer of the tree each sample is either unused or assigned to one of the nodes.
    When splitting a node, we need to know which samples are assigned to that node
    and then sort them with respect to each of the variables used.

    Thus we define the status of a sample in layer d as
        0 if the sample is not assigned to any node in the layer
        k + 1 if the sample is assigned to node number k in the layer.

    Note that for layer 0 (i.e. the root) the status of every sample is 0 or 1;
    this observation is used in the implementation of initSampleStatus_(), initTree_(), and initOrderedSamples_().

    In the constructor we sort all samples with respect to each variable once and for all.
    No further sorting is done, we simply extract sorted sublists from these presorted lists.

    The main tasks carried out by the code are:
        1. Maintain a vector that contains the status of each sample in the current layer of the tree.
        2. Using 1, maintain a buffer (or a several buffers) where the samples are grouped according node (or status)
            and then each group is sorted according to a variable.
            The buffer (or buffers) may or may not contain the unused samples (status = 0) as an initial group.
        3. Using 2, find the best split, if any, of each node in the current layer.

Short variable names:

    d depth (used to index the layers in the tree, d = 0 being the root)
    i sample index
    j variable index
    k node and node trainer index
    s sample status
    t0, t1, t2 threadlocal data
    w weight
    x indata value
    y outdata value
    z sample stratum

    Optimizations:
        branchfree code (not std::copy_if)
        fast random number generator (not std::mt19937)
        fast Bernoulli distribution (not std::bernoulli_distribution)
        fast sorting algorithm (not std::sort)
        memory optimized storage of vectors of sample indices and sample status
            using the dynamically selected template parameters SampleIndex and SampleStatus
        very few memory allocations
            using thread local buffers that expand as needed and never shrink
            besides these buffers, the only memory allocations are in the constructor and in createPredictor_()
*/

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
TreeTrainerImpl<SampleIndex>::TreeTrainerImpl(CRefXXfc inData, CRefXu8 strata) :
    inData_{inData},
    sampleCount_{static_cast<size_t>(inData.rows())},
    variableCount_{static_cast<size_t>(inData.cols())},
    sortedSamples_{initSortedSamples_()},
    strata_{strata},
    stratumCount_{strata_.rows() == 0 ? static_cast<size_t>(0) : static_cast<size_t>(strata_.maxCoeff()) + 1},
    sampleCountsByStratum_(initSampleCountsByStratum())
{
}

// The next function creates a list of sorted samples for each variable.
// These lists are then used by initSortedSamplesFast_() and initSortedSamples_().

template<typename SampleIndex>
vector<vector<SampleIndex>> TreeTrainerImpl<SampleIndex>::initSortedSamples_() const
{
    vector<vector<SampleIndex>> sortedSamples(variableCount_);

    const size_t threadCount = std::min<size_t>(omp_get_max_threads(), variableCount_);

    BEGIN_OMP_PARALLEL(threadCount)
    {
        const size_t sampleCount = sampleCount_;
        vector<pair<float, SampleIndex>> tmp(sampleCount);

        const size_t threadId = omp_get_thread_num();
        const size_t jStart = variableCount_ * threadId / threadCount;
        const size_t jStop = variableCount_ * (threadId + 1) / threadCount;

        for (size_t j = jStart; j != jStop; ++j) {

            const float* pInDataColJ = std::data(inData_.col(j));
            for (size_t i = 0; i != sampleCount; ++i)
                tmp[i] = {pInDataColJ[i], static_cast<SampleIndex>(i)};
            pdqsort_branchless(begin(tmp), end(tmp), ::firstLess);

            sortedSamples[j].resize(sampleCount);
            SampleIndex* pSortedSamplesJ = data(sortedSamples[j]);
            for (size_t i = 0; i != sampleCount; ++i)
                pSortedSamplesJ[i] = tmp[i].second;
        }
    }
    END_OMP_PARALLEL

    return sortedSamples;
}


template<typename SampleIndex>
vector<size_t> TreeTrainerImpl<SampleIndex>::initSampleCountsByStratum() const
{
    ASSERT(static_cast<size_t>(strata_.rows()) == sampleCount_);
    vector<size_t> n(stratumCount_, 0);
    for (size_t z : strata_)
        ++n[z];
    return n;
}

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
unique_ptr<BasePredictor> TreeTrainerImpl<SampleIndex>::trainImpl0_(
    CRefXd outData, CRefXd weights, const BaseOptions& options, size_t threadCount) const
{
    size_t ITEM_COUNT = 0;
    ScopedProfiler sp(PROFILE::TREE_TRAIN, &ITEM_COUNT);

    // PROFILE::SWITCH(PROFILE::VALIDATE_DATA, ITEM_COUNT);
    // validateData_(CRefXd outData, CRefXd weights);
    // ITEM_COUNT = sampleCount_;

    initTree_();

    const size_t usedVariableCount = usedVariableCount_(options);

    if (threadCount == 0 || threadCount > omp_get_max_threads())
        threadCount = omp_get_max_threads();

    const TrainData_ trainData{outData, weights, options, usedVariableCount, threadCount};

    // The current status of a sample is 0 if it is unused and k + 1 (with k = 0, 1, ..., n - 1) if it belongs to node
    // k. Here n is the number of nodes in the current layer of the tree. Thus 0 <= status <= the largest number of
    // nodes in any layer of the tree. The sample status is stored as type SampleStatus which we choose as the narrowest
    // possible unsigned integer type.

    const size_t maxNodeCount   // max node count in any layer of the tree
        = std::min<size_t>(1LL << options.maxTreeDepth(), std::max<size_t>(1, sampleCount_ / options.minNodeSize()));

    if (maxNodeCount <= 0xff) {
        using SampleStatus = uint8_t;
        ITEM_COUNT = trainImpl1_<SampleStatus>(&trainData, ITEM_COUNT);
    }
    else if (maxNodeCount <= 0xffff) {
        using SampleStatus = uint16_t;
        ITEM_COUNT = trainImpl1_<SampleStatus>(&trainData, ITEM_COUNT);
    }
    else if (maxNodeCount <= 0xffffffff) {
        using SampleStatus = uint32_t;
        ITEM_COUNT = trainImpl1_<SampleStatus>(&trainData, ITEM_COUNT);
    }
    else {
        using SampleStatus = uint64_t;
        ITEM_COUNT = trainImpl1_<SampleStatus>(&trainData, ITEM_COUNT);
    }

    PROFILE::SWITCH(PROFILE::FINALIZE_TREE, ITEM_COUNT);
    ITEM_COUNT = 0;

    ThreadLocalData0_& t0 = threadLocalData0_;
    const TreeNodeExt* root = data(t0.tree.front());
    return TreePredictor::createInstance(root);
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::validateData_(CRefXd outData, CRefXd weights) const
{
    if (static_cast<size_t>(outData.rows()) != sampleCount_)
        throw std::invalid_argument("Train indata and outdata have different numbers of samples.");
    if (!outData.isFinite().all())
        throw std::invalid_argument("Train outdata has values that are infinity or NaN.");

    if (static_cast<size_t>(weights.rows()) != sampleCount_)
        throw std::invalid_argument("Train indata and weights have different numbers of samples.");
    if (!weights.isFinite().all())
        throw std::invalid_argument("Train weights have values that are infinity or NaN.");
    if (!(weights >= 0.0).all())
        throw std::invalid_argument("Train weights have non-negative values.");
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initTree_() const
{
    ThreadLocalData0_& t0 = threadLocalData0_;

    // create root if needed
    if (empty(t0.tree)) {
        t0.tree.resize(1);
        t0.tree.front().resize(1);
    }

    TreeNodeExt* root = data(t0.tree.front());
    root->isLeaf = true;
}


template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::usedVariableCount_(const BaseOptions& options) const
{
    const size_t candidateVariableCount = std::min(variableCount_, options.topVariableCount());
    const size_t usedVariableCount
        = static_cast<size_t>(std::round(options.usedVariableRatio() * candidateVariableCount));
    return usedVariableCount;
}

//.......................................................................................................................

template<typename SampleIndex>
template<typename SampleStatus>
size_t TreeTrainerImpl<SampleIndex>::trainImpl1_(const TrainData_* trainData, size_t ITEM_COUNT) const
{
    PROFILE::SWITCH(PROFILE::ZERO, ITEM_COUNT);   // calibrate the profiling
    ITEM_COUNT = 0;

    PROFILE::SWITCH(PROFILE::INIT_SAMPLE_STATUS, ITEM_COUNT);
    ITEM_COUNT = sampleCount_;
    size_t usedSampleCount = initSampleStatus_<SampleStatus>(trainData);

    if (trainData->usedVariableCount == 0)
        // initsampleStatus_() sets y in the root
        return ITEM_COUNT;

    for (size_t d = 0; d != trainData->options.maxTreeDepth(); ++d) {

        if (d == 0 || trainData->options.selectVariablesByLevel()) {
            PROFILE::SWITCH(PROFILE::USED_VARIABLES, ITEM_COUNT);
            ITEM_COUNT = initUsedVariables_(trainData);
        }

        PROFILE::SWITCH(PROFILE::TREE_TRAIN, ITEM_COUNT);
        ITEM_COUNT = 0;

        initNodeTrainers_(trainData, d);   // very fast, no need to profile

        ITEM_COUNT = updateNodeTrainers1_<SampleStatus>(trainData, d, usedSampleCount, ITEM_COUNT);

        PROFILE::SWITCH(PROFILE::FINALIZE_SPLITS, ITEM_COUNT);
        ITEM_COUNT = 0;
        usedSampleCount = finalizeNodeTrainers_(trainData, d);   // very fast, no need to profile

        if (usedSampleCount == 0)
            break;
        if (d + 1 == trainData->options.maxTreeDepth())
            break;

        PROFILE::SWITCH(PROFILE::UPDATE_SAMPLE_STATUS, ITEM_COUNT);
        ITEM_COUNT = sampleCount_;
        updateSampleStatus_<SampleStatus>(trainData, d);

    }   // end d loop

    // TO DO: PRUNE TREE

    return ITEM_COUNT;
};


// The next function randomly assigns status 0 (unused) or 1 (belongs to the root) to each sample
// It also calculates nodeCount, sumW, sumWY and y for the root

template<typename SampleIndex>
template<typename SampleStatus>
size_t TreeTrainerImpl<SampleIndex>::initSampleStatus_(const TrainData_* trainData) const
{
    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& t1 = threadLocalData1_<SampleIndex>;
    ThreadLocalData2_<SampleStatus>& t2 = threadLocalData2_<SampleStatus>;
    RandomNumberEngine& rne = ::theRne;

    const size_t sampleCount = sampleCount_;
    t2.sampleStatus.resize(sampleCount);
    SampleStatus* pSampleStatus = data(t2.sampleStatus);

    const double minSampleWeight = (trainData->options.minRelSampleWeight() == 0)
                                       ? trainData->options.minAbsSampleWeight()
                                       : std::max(
                                           trainData->options.minAbsSampleWeight(),
                                           trainData->options.minRelSampleWeight() * trainData->weights.maxCoeff());

    if (minSampleWeight == 0.0) {

        if (trainData->options.usedSampleRatio() == 1.0) {
            // select all samples
            std::fill(pSampleStatus, pSampleStatus + sampleCount, static_cast<SampleStatus>(1));
        }

        else if (!trainData->options.stratifiedSamples()) {

            size_t n = sampleCount;

            // m = number of used samples
            size_t m = static_cast<size_t>(std::round(trainData->options.usedSampleRatio() * sampleCount));

            // randomly select m of n samples
            for (size_t i = 0; i != sampleCount; ++i) {
                const bool s = BernoulliDistribution_(m, n)(rne);
                pSampleStatus[i] = s;
                m -= s;
                --n;
            }
            ASSERT(m == 0);
            // for this assertion, and other similar assertions below, to hold it is critical that
            // BernoulliDistribution(0, n) with n > 0 always returns false and that
            // BernoulliDistribution(n, n) with n > 0 always returns true
        }

        else {
            const uint8_t* pStrata = std::data(strata_);

            // n[z] number of samples in stratum z
            // m[z] number of used samples in stratum z
            array<size_t, 256> n;
            array<size_t, 256> m;
            for (size_t z = 0; z != stratumCount_; ++z) {
                n[z] = sampleCountsByStratum_[z];
                m[z] = static_cast<size_t>(std::round(trainData->options.usedSampleRatio() * n[z]));
            }

            // for each z, randomly select m[z] of n[z] samples in stratum z
            for (size_t i = 0; i != sampleCount; ++i) {
                const uint8_t z = pStrata[i];
                const bool s = BernoulliDistribution_(m[z], n[z])(rne);
                pSampleStatus[i] = s;
                m[z] -= s;
                --n[z];
            }
            ASSERT(accumulate(begin(m), begin(m) + stratumCount_, static_cast<size_t>(0)) == 0);
        }
    }

    else {   // minSampleWeight > 0.0

        const double* pWeights = std::data(trainData->weights);

        if (trainData->options.usedSampleRatio() == 1.0) {
            // select all samples with weight >= min sample weight
            for (size_t i = 0; i != sampleCount; ++i) {
                const bool s = pWeights[i] >= minSampleWeight;
                pSampleStatus[i] = s;
            }
        }

        else if (!trainData->options.stratifiedSamples()) {

            // store all samples with weight >= min sample weight in sample buffer
            t1.sampleBuffer.resize(sampleCount);
            SampleIndex* p = data(t1.sampleBuffer);
            for (size_t i = 0; i != sampleCount; ++i) {
                *p = static_cast<SampleIndex>(i);
                const bool b = pWeights[i] >= minSampleWeight;
                p += b;
            }

            // n = number of samples with weight >= min sample weight
            size_t n = p - data(t1.sampleBuffer);
            t1.sampleBuffer.resize(n);

            // m = number of used samples
            size_t m = static_cast<size_t>(std::round(trainData->options.usedSampleRatio() * n));

            // randomly select m of the n samples with weight >= min sample weight
            std::fill(pSampleStatus, pSampleStatus + sampleCount, static_cast<SampleStatus>(0));
            for (SampleIndex i : t1.sampleBuffer) {
                const bool s = BernoulliDistribution_(m, n)(rne);
                pSampleStatus[i] = s;
                m -= s;
                --n;
            }
            ASSERT(m == 0);
        }

        else {
            const uint8_t* pStrata = std::data(strata_);

            // store all samples with weight >= min sample weight in sample buffer
            // n[z] = number of samples in stratum z with weight >= min sample weight
            array<size_t, 256> n;
            for (size_t z = 0; z != stratumCount_; ++z)
                n[z] = 0;
            t1.sampleBuffer.resize(sampleCount);
            SampleIndex* p = data(t1.sampleBuffer);
            for (size_t i = 0; i != sampleCount; ++i) {
                *p = static_cast<SampleIndex>(i);
                const bool b = pWeights[i] >= minSampleWeight;
                p += b;
                const uint8_t z = pStrata[i];
                n[z] += b;
            }
            t1.sampleBuffer.resize(p - data(t1.sampleBuffer));

            // m[z] = number of used samples in stratum z
            array<size_t, 256> m;
            for (size_t z = 0; z != stratumCount_; ++z)
                m[z] = static_cast<size_t>(std::round(trainData->options.usedSampleRatio() * n[z]));

            // for each z, randomly select m[z] of n[z] samples with weight >= min sample weight in stratum z
            std::fill(pSampleStatus, pSampleStatus + sampleCount, static_cast<SampleStatus>(0));
            for (SampleIndex i : t1.sampleBuffer) {
                const uint8_t z = pStrata[i];
                const bool s = BernoulliDistribution_(m[z], n[z])(rne);
                pSampleStatus[i] = s;
                m[z] -= s;
                --n[z];
            }
            ASSERT(accumulate(begin(m), begin(m) + stratumCount_, static_cast<size_t>(0)) == 0);
        }

    }   // end minSampleWeight == 0.0


    // init sums

    size_t usedSampleCount = 0;
    double sumW = 0.0;
    double sumWY = 0.0;

    const double* pOutData = std::data(trainData->outData);
    const double* pWeights = std::data(trainData->weights);
    for (size_t i = 0; i != sampleCount; ++i) {
        SampleStatus s = pSampleStatus[i];
        const double w = pWeights[i];
        const double y = pOutData[i];
        usedSampleCount += s;
        sumW += s * w;
        sumWY += s * w * y;
    }

    TreeNodeExt* root = data(t0.tree.front());
    root->sampleCount = usedSampleCount;
    root->sumW = sumW;
    root->sumWY = sumWY;
    root->y = (sumW == 0.0) ? 0.0f : static_cast<float>(sumWY / sumW);

    return usedSampleCount;
}


// The next function updates the status of each sample based on layer d to the status based on layer d + 1.
// It also recalculates nodeCount, sumW, sumWY and y with better precision for each node in layer d + 1.

template<typename SampleIndex>
template<typename SampleStatus>
void TreeTrainerImpl<SampleIndex>::updateSampleStatus_(const TrainData_* trainData, size_t d) const
{
    const size_t threadCount = trainData->threadCount;
    if (threadCount == 1)
        updateSampleStatusNoThreads_<SampleStatus>(trainData, d);
    else
        updateSampleStatusThreaded_<SampleStatus>(trainData, d, threadCount);
}


template<typename SampleIndex>
template<typename SampleStatus>
void TreeTrainerImpl<SampleIndex>::updateSampleStatusNoThreads_(const TrainData_* trainData, size_t d) const
{
    const double* pOutData = std::data(trainData->outData);
    const double* pWeights = std::data(trainData->weights);

    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData2_<SampleStatus>& t2 = threadLocalData2_<SampleStatus>;

    const vector<TreeNodeExt>& parentNodes = t0.tree[d];
    const TreeNodeExt* pParentNodes = data(parentNodes);
    vector<TreeNodeExt>& childNodes = t0.tree[d + 1];
    TreeNodeExt* pChildNodes = data(childNodes);

    SampleStatus* pSampleStatus = data(t2.sampleStatus);

    for (TreeNodeExt& childNode : childNodes) {
        childNode.sampleCount = 0;
        childNode.sumW = 0.0;
        childNode.sumWY = 0.0;
    }

    const size_t sampleCount = sampleCount_;
    for (size_t i = 0; i != sampleCount; ++i) {
        const SampleStatus s1 = pSampleStatus[i];
        if (s1 == 0)
            continue;
        const TreeNodeExt* pParentNode = &pParentNodes[s1 - 1];
        if (pParentNode->isLeaf) {
            pSampleStatus[i] = 0;
            continue;
        }
        TreeNodeExt* pChildNode = (inData_(i, pParentNode->j) < pParentNode->x)
                                      ? static_cast<TreeNodeExt*>(pParentNode->leftChild)
                                      : static_cast<TreeNodeExt*>(pParentNode->rightChild);
        const SampleStatus s2 = static_cast<SampleStatus>((pChildNode - pChildNodes) + 1);
        pSampleStatus[i] = s2;

        ++pChildNode->sampleCount;
        const double w = pWeights[i];
        const double y = pOutData[i];
        pChildNode->sumW += w;
        pChildNode->sumWY += w * y;
    }

    for (TreeNodeExt& childNode : childNodes)
        childNode.y = (childNode.sumW == 0) ? 0.0f : static_cast<float>(childNode.sumWY / childNode.sumW);
}


template<typename SampleIndex>
template<typename SampleStatus>
void TreeTrainerImpl<SampleIndex>::updateSampleStatusThreaded_(
    const TrainData_* trainData, size_t d, size_t threadCount) const
{

    const double* pOutData = std::data(trainData->outData);
    const double* pWeights = std::data(trainData->weights);

    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData2_<SampleStatus>& t2 = threadLocalData2_<SampleStatus>;

    const vector<TreeNodeExt>& parentNodes = t0.tree[d];
    const TreeNodeExt* pParentNodes = data(parentNodes);
    vector<TreeNodeExt>& childNodes = t0.tree[d + 1];
    size_t childNodeCount = size(childNodes);
    TreeNodeExt* pChildNodes = data(childNodes);

    SampleStatus* pSampleStatus = data(t2.sampleStatus);

    size_t padding = ::divideRoundUp(std::hardware_destructive_interference_size, sizeof(TreeNodeData));
    t0.treeData.assign(threadCount * (childNodeCount + padding), {0, 0.0, 0.0});

    BEGIN_OMP_PARALLEL(threadCount)
    {
        const size_t threadIndex = omp_get_thread_num();
        const size_t sampleCount = sampleCount_;
        const size_t iStart = sampleCount * threadIndex / threadCount;
        const size_t iStop = sampleCount * (threadIndex + 1) / threadCount;

        TreeNodeData* pTreeData = data(t0.treeData) + threadIndex * (childNodeCount + padding);

        for (size_t i = iStart; i != iStop; ++i) {
            const SampleStatus s1 = pSampleStatus[i];
            if (s1 == 0)
                continue;
            const TreeNodeExt* pParentNode = &pParentNodes[s1 - 1];
            if (pParentNode->isLeaf) {
                pSampleStatus[i] = 0;
                continue;
            }
            const TreeNodeExt* pChildNode = (inData_(i, pParentNode->j) < pParentNode->x)
                                                ? static_cast<TreeNodeExt*>(pParentNode->leftChild)
                                                : static_cast<TreeNodeExt*>(pParentNode->rightChild);
            const SampleStatus s2 = static_cast<SampleStatus>((pChildNode - pChildNodes) + 1);
            pSampleStatus[i] = s2;

            TreeNodeData* pNodeData = pTreeData + (s2 - 1);
            ++pNodeData->sampleCount;
            const double w = pWeights[i];
            const double y = pOutData[i];
            pNodeData->sumW += w;
            pNodeData->sumWY += w * y;
        }
    }
    END_OMP_PARALLEL

    for (TreeNodeExt& childNode : childNodes) {
        childNode.sampleCount = 0;
        childNode.sumW = 0.0;
        childNode.sumWY = 0.0;
    }
    for (size_t threadIndex = 0; threadIndex != threadCount; ++threadIndex) {
        const TreeNodeData* pTreeData = data(t0.treeData) + threadIndex * (childNodeCount + padding);
        for (size_t k = 0; k != childNodeCount; ++k) {
            pChildNodes[k].sampleCount += pTreeData[k].sampleCount;
            pChildNodes[k].sumW += pTreeData[k].sumW;
            pChildNodes[k].sumWY += pTreeData[k].sumWY;
        }
    }
    for (TreeNodeExt& childNode : childNodes)
        childNode.y = (childNode.sumW == 0) ? 0.0f : static_cast<float>(childNode.sumWY / childNode.sumW);
}


template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::initUsedVariables_(const TrainData_* trainData) const
{
    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& t1 = threadLocalData1_<SampleIndex>;
    RandomNumberEngine& rne = ::theRne;

    const size_t candidateVariableCount = std::min(variableCount_, trainData->options.topVariableCount());

    t0.usedVariables.resize(trainData->usedVariableCount);
    size_t* pUsedVariables = data(t0.usedVariables);

    size_t j = 0;
    size_t n = candidateVariableCount;
    size_t m = trainData->usedVariableCount;
    while (m != 0) {
        *pUsedVariables = j;
        const bool b = BernoulliDistribution_(m, n)(rne);
        pUsedVariables += b;
        m -= b;
        --n;
        ++j;
    }

    if (!trainData->options.saveMemory() && !trainData->options.selectVariablesByLevel()
        && trainData->options.maxTreeDepth() != 1)
        t1.orderedSamplesByVariable.resize(std::max(trainData->usedVariableCount, size(t1.orderedSamplesByVariable)));

    return j;   // number of iterations of the loop
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initNodeTrainers_(const TrainData_* trainData, size_t d) const
{
    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& t1 = threadLocalData1_<SampleIndex>;

    const vector<TreeNodeExt>& parentNodes = t0.tree[d];
    const size_t parentNodeCount = size(parentNodes);
    t1.treeNodeTrainers.resize(trainData->threadCount * parentNodeCount);

    for (size_t k = 0; k != parentNodeCount; ++k)
        t1.treeNodeTrainers[k].init(parentNodes[k], trainData->options);
    for (size_t threadIndex = 1; threadIndex != trainData->threadCount; ++threadIndex) {
        const size_t k0 = threadIndex * parentNodeCount;
        for (size_t k = 0; k != parentNodeCount; ++k)
            t1.treeNodeTrainers[k].fork(&t1.treeNodeTrainers[k0 + k]);
    }
}


// The next function combines the best split found by each thread to a single best split for each node.
// Then it creates layer d + 1 in the tree based on the best split for each node in layer d.
// It also calculates y for each node in the new layer.
// It returns the number of samples used by the nodes in the new layer.

template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::finalizeNodeTrainers_(const TrainData_* trainData, size_t d) const
{
    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& t1 = threadLocalData1_<SampleIndex>;

    t0.tree.resize(std::max(d + 2, size(t0.tree)));
    vector<TreeNodeExt>& parentNodes = t0.tree[d];
    const size_t parentNodeCount = size(parentNodes);

    for (size_t threadIndex = 1; threadIndex != trainData->threadCount; ++threadIndex) {
        const size_t k0 = threadIndex * parentNodeCount;
        for (size_t k = 0; k != parentNodeCount; ++k)
            t1.treeNodeTrainers[k].join(t1.treeNodeTrainers[k0 + k]);
    }

    vector<TreeNodeExt>& childNodes = t0.tree[d + 1];
    const size_t maxChildNodeCount = 2 * parentNodeCount;
    childNodes.resize(maxChildNodeCount);

    size_t usedSampleCount = 0;
    TreeNodeExt* pParentNode = data(parentNodes);
    TreeNodeExt* pChildNode = data(childNodes);
    for (size_t k = 0; k != parentNodeCount; ++k)
        usedSampleCount += t1.treeNodeTrainers[k].finalize(&pParentNode, &pChildNode);

    const size_t childNodeCount = pChildNode - data(childNodes);
    childNodes.resize(childNodeCount);

    return usedSampleCount;
}

//......................................................................................................................

template<typename SampleIndex>
template<typename SampleStatus>
size_t TreeTrainerImpl<SampleIndex>::updateNodeTrainers1_(
    const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t ITEM_COUNT) const
{
    const size_t threadCount = std::min(trainData->threadCount, std::max<size_t>(1, trainData->usedVariableCount));
    if (threadCount == 1)
        ITEM_COUNT = updateNodeTrainers1Nothreads_<SampleStatus>(trainData, d, usedSampleCount, ITEM_COUNT);
    else
        ITEM_COUNT = updateNodeTrainers1Threaded_<SampleStatus>(trainData, d, usedSampleCount, threadCount, ITEM_COUNT);
    return ITEM_COUNT;
}


template<typename SampleIndex>
template<typename SampleStatus>
size_t TreeTrainerImpl<SampleIndex>::updateNodeTrainers1Nothreads_(
    const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t ITEM_COUNT) const
{
    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& t1 = threadLocalData1_<SampleIndex>;
    ThreadLocalData2_<SampleStatus>& t2 = threadLocalData2_<SampleStatus>;

    t0.parent = &t0;
    t1.parent = &t1;
    t2.parent = &t2;

    const size_t threadIndex = 0;

    for (size_t usedVariableIndex = 0; usedVariableIndex != trainData->usedVariableCount; ++usedVariableIndex)
        ITEM_COUNT = updateNodeTrainers2_<SampleStatus>(
            trainData, d, usedSampleCount, usedVariableIndex, threadIndex, ITEM_COUNT);

    t0.parent = nullptr;
    t1.parent = nullptr;
    t2.parent = nullptr;

    return ITEM_COUNT;
}

template<typename SampleIndex>
template<typename SampleStatus>
size_t TreeTrainerImpl<SampleIndex>::updateNodeTrainers1Threaded_(
    const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t threadCount, size_t ITEM_COUNT) const
{
    PROFILE::SWITCH(PROFILE::INNER_THREAD_SYNCH, ITEM_COUNT);
    ITEM_COUNT = 0;

    ThreadLocalData0_& outerT0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& outerT1 = threadLocalData1_<SampleIndex>;
    ThreadLocalData2_<SampleStatus>& outerT2 = threadLocalData2_<SampleStatus>;


    std::atomic<size_t> nextUsedVariableIndex = 0;
    // std::cout << threadCount << std::endl;
    BEGIN_OMP_PARALLEL(threadCount)
    {
        size_t INNER_ITEM_COUNT = 0;

        PROFILE::SWITCH(PROFILE::TREE_TRAIN, INNER_ITEM_COUNT);
        INNER_ITEM_COUNT = 0;

        const size_t threadIndex = omp_get_thread_num();

        ThreadLocalData0_& innerT0 = threadLocalData0_;
        ThreadLocalData1_<SampleIndex>& innerT1 = threadLocalData1_<SampleIndex>;
        ThreadLocalData2_<SampleStatus>& innerT2 = threadLocalData2_<SampleStatus>;

        // give the inner threads access to the thread local data of the outer thread
        innerT0.parent = &outerT0;
        innerT1.parent = &outerT1;
        innerT2.parent = &outerT2;

        while (true) {
            const size_t usedVariableIndex = nextUsedVariableIndex++;
            if (usedVariableIndex >= trainData->usedVariableCount)
                break;

            INNER_ITEM_COUNT = updateNodeTrainers2_<SampleStatus>(
                trainData, d, usedSampleCount, usedVariableIndex, threadIndex, INNER_ITEM_COUNT);

            PROFILE::SWITCH(PROFILE::TREE_TRAIN, INNER_ITEM_COUNT);
            INNER_ITEM_COUNT = 0;
        }

        innerT0.parent = nullptr;
        innerT1.parent = nullptr;
        innerT2.parent = nullptr;

        PROFILE::SWITCH(PROFILE::INNER_THREAD_SYNCH, INNER_ITEM_COUNT);
        INNER_ITEM_COUNT = 0;
    }
    END_OMP_PARALLEL
    // std::cout << '.' << std::endl;
    ITEM_COUNT = 0;

    return ITEM_COUNT;
}


template<typename SampleIndex>
template<typename SampleStatus>
size_t TreeTrainerImpl<SampleIndex>::updateNodeTrainers2_(
    const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t usedVariableIndex, size_t threadIndex,
    size_t ITEM_COUNT) const
{
    if (d == 0) {
        PROFILE::SWITCH(PROFILE::INIT_ORDERED_SAMPLES, ITEM_COUNT);
        ITEM_COUNT = sampleCount_;
        initOrderedSamples_<SampleStatus>(trainData, d, usedSampleCount, usedVariableIndex);
    }
    else if (trainData->options.saveMemory() || trainData->options.selectVariablesByLevel()) {
        PROFILE::SWITCH(PROFILE::UPDATE_ORDERED_SAMPLES, ITEM_COUNT);
        ITEM_COUNT = sampleCount_;
        updateOrderedSampleSaveMemory_<SampleStatus>(trainData, d, usedSampleCount, usedVariableIndex);
    }
    else {
        PROFILE::SWITCH(PROFILE::UPDATE_ORDERED_SAMPLES, ITEM_COUNT);
        ITEM_COUNT = usedSampleCount;
        updateOrderedSamples_<SampleStatus>(trainData, d, usedSampleCount, usedVariableIndex);
    }

    PROFILE::SWITCH(PROFILE::UPDATE_SPLITS, ITEM_COUNT);
    ITEM_COUNT = usedSampleCount;
    updateNodeTrainers3_(trainData, d, usedVariableIndex, threadIndex);

    return ITEM_COUNT;
}


// The following three functions update the vectors t1.sampleBuffer and t1.orderedSampleBlocks.
// The vector t1.sampleBuffer contains all samples that are used in layer d of the tree.
// The vector is divided into blocks. For each  k = 0, 1, ..., nodeCount-1,
// block number k contains the samples that belong to node k in layer d of the tree.
// Each block is sorted according to variable j.
// The vector t1.orderedSampleBlocks contains pointers to the beginning of each block.
// This vector is then used by updateNodeTrainers_().
//
// The function initOrderedSamples_ does this for layer 0, i.e. the root.
// It copies samples from sortedSamples_[j] which contains all samples sorted according to variable j,
// discards the unused samples and places the used ones in a single block
//
// The function initOrderedSamples_ does this for any layer of depth d >= 0.
// It also copies samples from sortedSamples_[j] and distrubtes them in order in several blocks.
// In addition to one block for each node, it also creates an initial block with the unused samples.
//
// The function updateOrderedSamples_() also does this does for any layer of depth >= 1.
// But instead of copying the samples from sortedSamples_[j] it uses the ordered samples for the previous layer.
// It discards the unused samples and distributes the used samples in several blocks, one for each node.
// This is faster but requires more memory and only works if the same variables are used for each layer of the tree.
//
// (The code uses a branchfree conditional copy implementation that may overwrite the blocks by one element.
// Therefore we adding a dummy element after the sample block in initOrderedSamples_()
// and a dummy element after each block in updateOrderedSamples_().)


template<typename SampleIndex>
template<typename SampleStatus>
void TreeTrainerImpl<SampleIndex>::initOrderedSamples_(
    const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t usedVariableIndex) const
{
    // called from the inner threads so be careful to distinguish between t0 and t0.parent etc.

    ASSERT(d == 0);

    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& t1 = threadLocalData1_<SampleIndex>;
    ThreadLocalData2_<SampleStatus>& t2 = threadLocalData2_<SampleStatus>;

    const size_t j = t0.parent->usedVariables[usedVariableIndex];
    const SampleStatus* pSampleStatus = data(t2.parent->sampleStatus);

    t1.sampleBuffer.resize(usedSampleCount + 1 /*dummy*/);
    SampleIndex* pOrderedSamples = data(t1.sampleBuffer);

    for (SampleIndex i : sortedSamples_[j]) {
        *pOrderedSamples = i;
        const SampleStatus s = pSampleStatus[i];   // s = 0 (unused) or 1 (used, and hence belongs to the root)
        pOrderedSamples += s;
    }

    pOrderedSamples = data(t1.sampleBuffer);
    if (d + 1 != trainData->options.maxTreeDepth() && !trainData->options.saveMemory()
        && !trainData->options.selectVariablesByLevel())
        // save the ordered samples to be used as input when ordering samples for the next layer
        swap(t1.sampleBuffer, t1.parent->orderedSamplesByVariable[usedVariableIndex]);

    t1.orderedSampleBlocks.assign(1, pOrderedSamples);
}


template<typename SampleIndex>
template<typename SampleStatus>
void TreeTrainerImpl<SampleIndex>::updateOrderedSampleSaveMemory_(
    const TrainData_* /*trainData*/, size_t d, size_t usedSampleCount, size_t usedVariableIndex) const
{
    // called from the inner threads so be careful to distinguish between t0 and t0.parent etc.

    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& t1 = threadLocalData1_<SampleIndex>;
    ThreadLocalData2_<SampleStatus>& t2 = threadLocalData2_<SampleStatus>;

    const size_t j = t0.parent->usedVariables[usedVariableIndex];
    const SampleStatus* pSampleStatus = data(t2.parent->sampleStatus);

    const vector<TreeNodeExt>& nodes = t0.parent->tree[d];
    const size_t nodeCount = size(nodes);
    const size_t statusCount = nodeCount + 1;
    const size_t unusedSampleCount = sampleCount_ - usedSampleCount;

    t1.sampleBuffer.resize(sampleCount_);
    SampleIndex* pOrderedSamples = data(t1.sampleBuffer);

    t1.orderedSampleBlocks.resize(statusCount);
    SampleIndex** pOrderedSampleBlocks = data(t1.orderedSampleBlocks);

    for (size_t s = 0; s != statusCount; ++s) {
        pOrderedSampleBlocks[s] = pOrderedSamples;
        const size_t blockSize = (s == 0) ? unusedSampleCount : nodes[s - 1].sampleCount;
        pOrderedSamples += blockSize;
    }

    // orderedSampleBlocks[s] = pointer to the beginning of the block where we will store samples
    // with status s sorted according to variable j  (for each s = 0, 1, 2, ..., nodeCount)

    for (SampleIndex i : sortedSamples_[j]) {
        const SampleStatus s = pSampleStatus[i];
        *pOrderedSampleBlocks[s] = i;
        ++pOrderedSampleBlocks[s];
    }

    // orderedSampleBlocks[s] = pointer to the end of the block with samples with s
    // = beginning of the status s + 1 block = beginning of the block with samples that belong to node s
}


template<typename SampleIndex>
template<typename SampleStatus>
void TreeTrainerImpl<SampleIndex>::updateOrderedSamples_(
    const TrainData_* trainData, size_t d, size_t usedSampleCount, size_t usedVariableIndex) const
{
    // called from inner threads; be careful to distinguish between t0 and t0.parent etc.

    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& t1 = threadLocalData1_<SampleIndex>;
    ThreadLocalData2_<SampleStatus>& t2 = threadLocalData2_<SampleStatus>;

    const SampleIndex* pPrevOrderedSamples = data(t1.parent->orderedSamplesByVariable[usedVariableIndex]);
    const SampleStatus* pSampleStatus = data(t2.parent->sampleStatus);

    const vector<TreeNodeExt>& prevNodes = t0.parent->tree[d - 1];
    const vector<TreeNodeExt>& nodes = t0.parent->tree[d];
    const size_t nodeCount = size(nodes);

    t1.sampleBuffer.resize(usedSampleCount + nodeCount /*dummies*/);
    SampleIndex* pOrderedSamplesLeft = data(t1.sampleBuffer);

    for (const TreeNodeExt& prevNode : prevNodes) {

        if (prevNode.isLeaf) {
            pPrevOrderedSamples += prevNode.sampleCount + 1 /*dummy*/;
            continue;
        }

        const TreeNodeExt* leftNode = static_cast<const TreeNodeExt*>(prevNode.leftChild);
        SampleIndex* pOrderedSamplesRight = pOrderedSamplesLeft + leftNode->sampleCount + 1 /*dummy*/;

        const SampleIndex* pPrevOrderedSamplesEnd = pPrevOrderedSamples + prevNode.sampleCount;
        while (pPrevOrderedSamples != pPrevOrderedSamplesEnd) {
            const SampleIndex i = *pPrevOrderedSamples;
            ++pPrevOrderedSamples;
            *pOrderedSamplesLeft = i;
            *pOrderedSamplesRight = i;
            const SampleStatus s = pSampleStatus[i];
            pOrderedSamplesLeft += s & 1;
            pOrderedSamplesRight += 1 - s & 1;
        }

        pPrevOrderedSamples += 1 /*dummy*/;
        pOrderedSamplesLeft = pOrderedSamplesRight + 1 /*dummy*/;
    }

    SampleIndex* pOrderedSamples = data(t1.sampleBuffer);

    if (d + 1 != trainData->options.maxTreeDepth())
        // save the ordered samples to be used as input when ordering samples for the next layer
        swap(t1.parent->orderedSamplesByVariable[usedVariableIndex], t1.sampleBuffer);

    t1.orderedSampleBlocks.resize(nodeCount);
    SampleIndex** pOrderedSampleBlocks = data(t1.orderedSampleBlocks);
    for (size_t k = 0; k != nodeCount; ++k) {
        pOrderedSampleBlocks[k] = pOrderedSamples;
        const size_t blockSize = nodes[k].sampleCount;
        pOrderedSamples += blockSize + 1 /*dummy*/;
    }
}


// The next function determines the best split of each node in layer d with respect to variable j.
// The pointer pOrderedSamples points to a buffer that contains all used samples grouped by node
// and then sorted by variable j.

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateNodeTrainers3_(
    const TrainData_* trainData, size_t d, size_t usedVariableIndex, size_t threadIndex) const
{
    // called from inner threads; be careful to distinguish between t0 and t0.parent etc.

    ThreadLocalData0_& t0 = threadLocalData0_;
    ThreadLocalData1_<SampleIndex>& t1 = threadLocalData1_<SampleIndex>;

    const vector<TreeNodeExt>& parentNodes = t0.parent->tree[d];
    const size_t parentNodeCount = size(parentNodes);
    const size_t k0 = threadIndex * parentNodeCount;
    const SampleIndex* const* pOrderedSampleBlocks = data(t1.orderedSampleBlocks);
    const size_t j = t0.parent->usedVariables[usedVariableIndex];

    for (size_t k = 0; k != parentNodeCount; ++k) {

        const SampleIndex* pOrderedSampleBlockBegin = pOrderedSampleBlocks[k];
        const SampleIndex* pOrderedSampleBlockEnd = pOrderedSampleBlockBegin + parentNodes[k].sampleCount;

        t1.parent->treeNodeTrainers[k0 + k].update(
            inData_, trainData->outData, trainData->weights, pOrderedSampleBlockBegin, pOrderedSampleBlockEnd, j);
    }
}

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImpl<uint8_t>;
template class TreeTrainerImpl<uint16_t>;
template class TreeTrainerImpl<uint32_t>;
template class TreeTrainerImpl<uint64_t>;

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
    vector<pair<float, SampleIndex>> tmp{ sampleCount_ };
    for (size_t j = 0; j != variableCount_; ++j) {
        for (size_t i = 0; i != sampleCount_; ++i)
            tmp[i] = { inData_(i,j), static_cast<SampleIndex>(i) };
        pdqsort_branchless(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return x.first < y.first; });
        sortedSamples[j].resize(sampleCount_);
        for (size_t i = 0; i != sampleCount_; ++i)
            sortedSamples[j][i] = tmp[i].second;
    }
    return sortedSamples;
}

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
unique_ptr<BasePredictor> TreeTrainerImpl<SampleIndex>::train(
    CRefXd outData, CRefXd weights, const TreeOptions& options) const
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

    for (size_t d = 0; d != options.maxDepth(); ++d) {

        initSplits_(options, d);

        for (size_t usedVariableIndex = 0; usedVariableIndex != usedVariableCount; ++usedVariableIndex) {

            const SampleIndex* orderedSamples;
            if (options.altImplementation()) {
                if (d == 0) {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES);
                    ITEM_COUNT = sampleCount_;
                    orderedSamples = initOrderedSamplesAlt_(usedVariableIndex, usedSampleCount, d);
                }
                else {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_ORDERED_SAMPLES);
                    ITEM_COUNT = usedSampleCount;
                    orderedSamples = updateOrderedSamplesAlt_(usedVariableIndex, usedSampleCount, d);
                }
            }
            else {
                if (d == 0) {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::SET_ORDERED_SAMPLES_FAST);
                    ITEM_COUNT = sampleCount_;
                    orderedSamples = initOrderedSamplesFast_(usedVariableIndex, usedSampleCount, d);
                }
                else {
                    PROFILE::SWITCH(ITEM_COUNT, PROFILE::SET_ORDERED_SAMPLES);
                    ITEM_COUNT = sampleCount_;
                    orderedSamples = initOrderedSamples_(usedVariableIndex, usedSampleCount, d);
                }
            }

            PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SPLITS);
            ITEM_COUNT = usedSampleCount;
            updateSplits_(outData, weights, options, orderedSamples, usedVariableIndex, d);
        }

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_TREE);
        ITEM_COUNT = 0;
        size_t newNodeCount = finalizeSplits_(d);
        if (newNodeCount == 0) break;

        if (d + 1 == options.maxDepth()) break;     
        // if this test is removed, then y in the child nodes is recalculated
        // with more precision in updateSampleStatus_() 

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SAMPLE_STATUS);
        ITEM_COUNT = sampleCount_;
        usedSampleCount = updateSampleStatus_(outData, weights, d);
    }

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
    RandomNumberEngine_& rne = tlRne_;
    size_t candidateVariableCount = std::min(variableCount_, options.topVariableCount());
    size_t usedVariableCount = static_cast<size_t>(options.usedVariableRatio() * candidateVariableCount + 0.5);
    if (usedVariableCount == 0) usedVariableCount = 1;

    ASSERT(0 < usedVariableCount);
    ASSERT(usedVariableCount <= candidateVariableCount);
    ASSERT(candidateVariableCount <= variableCount_);

    tlUsedVariables_.resize(usedVariableCount);
    span<size_t> usedVariables{ tlUsedVariables_ };

    size_t n = candidateVariableCount;
    size_t k = usedVariableCount;
    size_t i = 0;
    auto p = begin(usedVariables);
    while (k > 0) {
        *p = i;
        bool b = BernoulliDistribution(k, n)(rne);
        p += b;
        k -= b;
        --n;
        ++i;
    }

    if (options.altImplementation())
        tlSampleBufferByVariable_.resize(std::max(usedVariableCount, size(tlSampleBufferByVariable_)));

    return std::make_pair(usedVariableCount, candidateVariableCount);
}


template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::initSampleStatus_(CRefXd weights0, const TreeOptions& options) const
{
    size_t usedSampleCount;

    RandomNumberEngine_& rne = tlRne_;

    tlSampleStatus_.resize(sampleCount_);
    span<SampleIndex> sampleStatus{ tlSampleStatus_ };

    const size_t minNodeSize = options.minNodeSize();

    double minSampleWeight = options.minAbsSampleWeight();
    double minRelSampleWeight = options.minRelSampleWeight();
    if (minRelSampleWeight > 0.0)
        minSampleWeight = std::max(minSampleWeight, weights0.maxCoeff() * minRelSampleWeight);

    if (minSampleWeight == 0.0) {

        if (!options.isStratified()) {

            // create a random mask of length n with k ones and n - k zeros
            // n = total number of samples
            // k = number of used samples

            size_t n = sampleCount_;
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0) k = 1;
            usedSampleCount = k;

            for (auto p = begin(sampleStatus); p != end(sampleStatus); ++p) {
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
            for (auto p = begin(sampleStatus); p != end(sampleStatus); ++p, ++q) {
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

        std::ranges::fill(sampleStatus, static_cast<SampleIndex>(0));

        tlSampleBuffer_.resize(sampleCount_);
        span<SampleIndex> sampleBuffer{ tlSampleBuffer_ };

        const double* weights = std::data(weights0);

        if (!options.isStratified()) {
            auto p = begin(sampleBuffer);
            for (size_t i = 0; i != sampleCount_; ++i) {
                bool b = weights[i] >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
            }
            tlSampleBuffer_.resize(p - begin(sampleBuffer));
            sampleBuffer = tlSampleBuffer_;

            size_t n = size(sampleBuffer);
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0 && n > 0) k = 1;
            usedSampleCount = k;

            for (SampleIndex i : sampleBuffer) {
                bool b = BernoulliDistribution(k, n) (rne);
                sampleStatus[i] = b;
                k -= b;
                --n;
            }
        }

        else {
            const size_t* strata = std::data(strata_);
            array<size_t, 2> n = { 0, 0 };
            auto p = begin(sampleBuffer);
            for (size_t i = 0; i != sampleCount_; ++i) {
                size_t stratum = strata[i];
                bool b = weights[i] >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
                n[stratum] += b;
            }
            tlSampleBuffer_.resize(p - begin(sampleBuffer));
            sampleBuffer = tlSampleBuffer_;

            array<size_t, 2> k;
            k[0] = static_cast<size_t>(options.usedSampleRatio() * n[0] + 0.5);
            if (k[0] == 0 && n[0] > 0) k[0] = 1;
            k[1] = static_cast<size_t>(options.usedSampleRatio() * n[1] + 0.5);
            if (k[1] == 0 && n[1] > 0) k[1] = 1;
            usedSampleCount = k[0] + k[1];

            for (size_t i : sampleBuffer) {
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
    double sumW = 0.0;
    double sumWY = 0.0;
    const SampleIndex* sampleStatus = data(tlSampleStatus_);
    const double* pOutData = &outData.coeffRef(0);
    const double* pWeights = &weights.coeffRef(0);
    for (size_t i = 0; i != sampleCount_; ++i) {
        double m = static_cast<double>(sampleStatus[i]);
        double w = pWeights[i];
        double y = pOutData[i];
        sumW += m * w;
        sumWY += m * w * y;
    }

    tlTree_.resize(std::max<size_t>(1, size(tlTree_)));
    tlTree_.front().resize(1);
    TreeNodeExt* root = data(tlTree_.front());
    root->isLeaf = true;
    root->sampleCount = usedSampleCount;
    root->sumW = sumW;
    root->sumWY = sumWY;
    root->y = sumW == 0.0 ? 0.0f : static_cast<float>(sumWY / sumW);
}

//......................................................................................................................

template<typename SampleIndex>
const SampleIndex* TreeTrainerImpl<SampleIndex>::initOrderedSamplesFast_(
    size_t usedVariableIndex, size_t usedSampleCount, size_t d) const
{
    span<TreeNodeExt> nodes{ tlTree_[d] };
    const size_t nodeCount = size(nodes);
    ASSERT(nodeCount == 1);

    span<SampleIndex> sampleStatus{ tlSampleStatus_ };

    size_t variableIndex = tlUsedVariables_[usedVariableIndex];
    span<const SampleIndex> sortedSamples{ sortedSamples_[variableIndex] };

    tlSampleBuffer_.resize(usedSampleCount);
    span<SampleIndex> sampleBuffer{ tlSampleBuffer_ };

    // branchfree code for copying all i in sortedSamples with sampleStatus[i] = 1 to sampleBuffer

    auto p = cbegin(sortedSamples);
    auto q = begin(sampleBuffer);
    auto qEnd = end(sampleBuffer);
    while (q != qEnd) {
        SampleIndex i = *p;
        *q = i;
        ++p;
        q += sampleStatus[i];
    }

    return data(sampleBuffer);
}


template<typename SampleIndex>
const SampleIndex* TreeTrainerImpl<SampleIndex>::initOrderedSamples_(
    size_t usedVariableIndex, size_t usedSampleCount, size_t d) const
{
    span<TreeNodeExt> nodes{ tlTree_[d] };
    const size_t nodeCount = size(nodes);

    span<const SampleIndex> sampleStatus = span(tlSampleStatus_);

    const size_t variableIndex = tlUsedVariables_[usedVariableIndex];
    span<const SampleIndex> sortedSamples{ sortedSamples_[variableIndex] };

    tlSampleBuffer_.resize(sampleCount_);
    span<SampleIndex> sampleBuffer = span(tlSampleBuffer_);

    tlSamplePointerBuffer_.resize(nodeCount + 1);
    span<SampleIndex*> samplePointerBuffer = span(tlSamplePointerBuffer_);

    SampleIndex* p = data(sampleBuffer);
    const size_t unusedSampleCount = sampleCount_ - usedSampleCount;
    for (size_t k = 0; k != nodeCount + 1; ++k) {
        samplePointerBuffer[k] = p;
        p += (k == 0 ? unusedSampleCount : nodes[k - 1].sampleCount);
    }

    // samplePointerBuffer[i] (i = 0, 1, 2, ... nodeCount) points to the block
    //  where we soon will store samples with status i 
    // (status i = 0 means unused sample, i > 0 means sample nbelongs to node number i - 1)

    for (SampleIndex i : sortedSamples) {
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
const SampleIndex* TreeTrainerImpl<SampleIndex>::initOrderedSamplesAlt_(
    size_t usedVariableIndex, size_t usedSampleCount, size_t d) const
{
    const SampleIndex* p = initOrderedSamplesFast_(usedVariableIndex, usedSampleCount, d);
    swap(tlSampleBuffer_, tlSampleBufferByVariable_[usedVariableIndex]);
    return p;
}


template<typename SampleIndex>
const SampleIndex* TreeTrainerImpl<SampleIndex>::updateOrderedSamplesAlt_(
    size_t usedVariableIndex, size_t usedSampleCount, size_t d) const
{
    span<TreeNodeExt> parentNodes{ tlTree_[d - 1] };

    span<TreeNodeExt> childNodes{ tlTree_[d] };
    size_t childNodeCount = size(childNodes);

    span<SampleIndex> sampleStatus{ tlSampleStatus_ };

    span<SampleIndex> prevSampleBuffer{ tlSampleBufferByVariable_[usedVariableIndex] };

    tlSampleBuffer_.resize(usedSampleCount);
    span<SampleIndex> sampleBuffer = span{ tlSampleBuffer_ };
        
    tlSamplePointerBuffer_.resize(childNodeCount + 1);
    span<SampleIndex*> samplePointerBuffer = span(tlSamplePointerBuffer_);

    SampleIndex* p = data(sampleBuffer);
    for (size_t k = 0; k != childNodeCount; ++k) {
        samplePointerBuffer[k + 1] = p;
        p += childNodes[k].sampleCount;
    }

    // samplePointerBuffer[i] (i = 1, 2, ... nodeCount-1) points to the block
    //  where we soon will store samples with status i 
    // (status i = 0 means unused sample, i > 0 means sample nbelongs to node number i - 1)

    p = data(prevSampleBuffer);
    for (const TreeNodeExt& parentNode: parentNodes) {
        SampleIndex* pEnd = p + parentNode.sampleCount;
        if (!parentNode.isLeaf)
            for (SampleIndex i : span(p, pEnd)) {
                SampleIndex s = sampleStatus[i];
                *(samplePointerBuffer[s]++) = i;
            }
        p = pEnd;
    }

    p = data(sampleBuffer);
    swap(tlSampleBufferByVariable_[usedVariableIndex], tlSampleBuffer_);
    return p;
}

//......................................................................................................................

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initSplits_(const TreeOptions& options, size_t d) const
{
    span<TreeNodeExt> parentNodes{ tlTree_[d] };
    size_t parentNodeCount = size(parentNodes);

    tlTreeNodeTrainers_.resize(parentNodeCount);
    span<TreeNodeTrainer<SampleIndex>> treeNodeTrainers{ tlTreeNodeTrainers_ };

    for (size_t k = 0; k != parentNodeCount; ++k)
        treeNodeTrainers[k].init(&parentNodes[k], options);
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateSplits_(
    CRefXd outData,
    CRefXd weights,
    const TreeOptions& options,
    const SampleIndex* orderedSamples,
    size_t usedVariableIndex,
    size_t d
) const
{
    span<TreeNodeExt> parentNodes{ tlTree_[d] };
    size_t parentNodeCount = size(parentNodes);

    span<TreeNodeTrainer<SampleIndex>> treeNodeTrainers{ tlTreeNodeTrainers_ };
    size_t variableIndex = tlUsedVariables_[usedVariableIndex];

    const SampleIndex* p0 = orderedSamples;
    for (size_t k = 0; k != parentNodeCount; ++k) {
        const SampleIndex* p1 = p0 + parentNodes[k].sampleCount;
        treeNodeTrainers[k].update(inData_, outData, weights, options, span(p0, p1), variableIndex);
        p0 = p1;
    }
}


template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::finalizeSplits_(size_t d) const
{
    span<TreeNodeExt> parentNodes{ tlTree_[d] };
    size_t parentNodeCount = size(parentNodes);

    size_t maxChildNodeCount = 2 * parentNodeCount;
    tlTree_.resize(std::max(d + 2, size(tlTree_)));
    tlTree_[d + 1].resize(maxChildNodeCount);
    span<TreeNodeExt> childNodes{ tlTree_[d + 1] };

    span<TreeNodeTrainer<SampleIndex>> treeNodeTrainers{ tlTreeNodeTrainers_ };
    TreeNodeExt* parentNode = data(parentNodes);
    TreeNodeExt* childNode = data(childNodes);
    for (size_t k = 0; k != parentNodeCount; ++k)
        treeNodeTrainers[k].finalize(&parentNode, &childNode);

    size_t childNodeCount = childNode - data(childNodes);
    tlTree_[d + 1].resize(childNodeCount);
    return childNodeCount;
}

//......................................................................................................................

template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::updateSampleStatus_(CRefXd outData, CRefXd weights, size_t d) const
{
    span<TreeNodeExt> parentNodes{ tlTree_[d] };
    span<TreeNodeExt> childNodes{ tlTree_[d + 1] };
    span<SampleIndex> sampleStatus{ tlSampleStatus_ };

    for (TreeNodeExt& childNode : childNodes) {
        childNode.sampleCount = 0;
        childNode.sumW = 0.0;
        childNode.sumWY = 0.0;
    }

    const double* pOutData = &outData.coeffRef(0);
    const double* pWeights = &weights.coeffRef(0);

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
    unique_ptr<BasePredictor> pred;
    const TreeNodeExt* root = data(tlTree_.front());
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

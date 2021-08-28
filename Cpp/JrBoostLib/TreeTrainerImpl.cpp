//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImpl.h"

#include "ExceptionSafeOmp.h"
#include "InterruptHandler.h"
#include "TreeOptions.h"
#include "StumpPredictor.h"
#include "TrivialPredictor.h"
#include "TreePredictor.h"
#include "pdqsort.h"

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
TreeTrainerImpl<SampleIndex>::TreeTrainerImpl(CRefXXf inData, CRefXs strata) :
    inData_{ inData },
    sampleCount_{ static_cast<size_t>(inData.rows()) },
    variableCount_{ static_cast<size_t>(inData.cols()) },
    sortedSamples_{ createSortedSamples_() },
    strata_{ strata },
    stratum0Count_{ (strata == 0).cast<size_t>().sum() },
    stratum1Count_{ (strata == 1).cast<size_t>().sum() }
{
}


template<typename SampleIndex>
vector<vector<SampleIndex>> TreeTrainerImpl<SampleIndex>::createSortedSamples_() const
{
    PROFILE::PUSH(PROFILE::STUMP_INIT);
    uint64_t ITEM_COUNT = static_cast<uint64_t>(
        sampleCount_ * std::log2(static_cast<double>(sampleCount_)) * variableCount_);

    vector<vector<SampleIndex>> sortedSamples(variableCount_);
    vector<pair<float, SampleIndex>> tmp{ sampleCount_ };
    for (size_t j = 0; j < variableCount_; ++j) {
        for (size_t i = 0; i < sampleCount_; ++i)
            tmp[i] = { inData_(i,j), static_cast<SampleIndex>(i) };
        pdqsort_branchless(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return x.first < y.first; });
        sortedSamples[j].resize(sampleCount_);
        for (size_t i = 0; i < sampleCount_; ++i)
            sortedSamples[j][i] = tmp[i].second;
    }

    PROFILE::POP(ITEM_COUNT);

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
    size_t usedSampleCount = initSampleStatus_(options, weights);

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_TREE);
    ITEM_COUNT = 0;
    initTree_(usedSampleCount, options);

    size_t d = 0;
    while (true) {

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SPLITS);
        ITEM_COUNT = 0;
        initSplits_(d);

        for (size_t usedVariableIndex = 0; usedVariableIndex != usedVariableCount; ++usedVariableIndex) {

            // sort the samples in each node in layer d according to the variable specifed by usedVariableIndex
            if (d == 0) {
                PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES);
                ITEM_COUNT = sampleCount_;
                if (options.altImplementation() && options.maxDepth() > 1)
                    initOrderedSamplesAlt_(usedVariableIndex, usedSampleCount);
                else
                    initOrderedSamples_(usedVariableIndex, usedSampleCount);
            }
            else {
                PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_ORDERED_SAMPLES);
                if (options.altImplementation()) {
                    ITEM_COUNT = usedSampleCount;
                    updateOrderedSamplesAlt_(usedVariableIndex, usedSampleCount, d);
                }
                else {
                    ITEM_COUNT = sampleCount_;
                    updateOrderedSamples_(usedVariableIndex, usedSampleCount);
                }
            }

            // find the best split for each node in layer d according to variable specified by usedVariableIndex
            PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SPLITS);
            ITEM_COUNT = usedSampleCount;
            updateSplits_(usedVariableIndex, outData, weights, options);
        }

        // CLEAN UP THIS CODE
        if (std::this_thread::get_id() == mainThreadId) {
            for (const Split& split: tlSplits_) {
                PROFILE::SPLIT_ITERATION_COUNT += split.iterationCount;
                PROFILE::SLOW_BRANCH_COUNT += split.slowBranchCount;
            }
        }

        // create layer d + 1 in the tree
        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_TREE);
        ITEM_COUNT = 0;
        usedSampleCount = updateTree_(d);

        ++d;
        if (d == options.maxDepth() || usedSampleCount == 0) break;
        // usedSampleCount == 0 if and only if layer d (note that d has been incremented!) in the tree is empty

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SAMPLE_STATUS);
        ITEM_COUNT = sampleCount_;
        updateSampleStatus_(d);
    }

    // TO DO: PRUNE TREE

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
    std::span usedVariables{ tlUsedVariables_ };


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

    if (options.altImplementation() && size(tlOrderedSamplesByVariable_) < usedVariableCount)
        tlOrderedSamplesByVariable_.resize(usedVariableCount);

    return std::make_pair(usedVariableCount, candidateVariableCount);
}


template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::initSampleStatus_(const TreeOptions& options, CRefXd weights) const
{
    RandomNumberEngine_& rne = tlRne_;
    const size_t minNodeSize = options.minNodeSize();

    size_t usedSampleCount;

    tlSampleStatus_.resize(sampleCount_);
    std::span sampleStatus{ tlSampleStatus_ };

    double minSampleWeight = options.minAbsSampleWeight();
    double minRelSampleWeight = options.minRelSampleWeight();
    if (minRelSampleWeight > 0.0)
        minSampleWeight = std::max(minSampleWeight, weights.maxCoeff() * minRelSampleWeight);

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

            const size_t* s = &strata_.coeffRef(0);
            for (auto p = begin(sampleStatus); p != end(sampleStatus); ++p) {
                size_t stratum = *s;
                ++s;
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
        std::span sampleBuffer{ tlSampleBuffer_ };

        if (!options.isStratified()) {
            auto p = begin(sampleBuffer);
            for (size_t i = 0; i < sampleCount_; ++i) {
                bool b = weights(i) >= minSampleWeight;
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
            array<size_t, 2> n = { 0, 0 };
            auto p = begin(sampleBuffer);
            for (size_t i = 0; i < sampleCount_; ++i) {
                size_t stratum = strata_[i];
                bool b = weights(i) >= minSampleWeight;
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
                size_t stratum = strata_[i];
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
void TreeTrainerImpl<SampleIndex>::initTree_(size_t usedSampleCount, const TreeOptions& options) const
{
    tlTree_.resize(std::max(options.maxDepth() + 1, size(tlTree_)));
    tlTree_.front().resize(1);

    tlSampleCountByParentNode_ = {};
    tlSampleCountByChildNode_ = { usedSampleCount };
}

//......................................................................................................................

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initSplits_(size_t d) const
{
    size_t nodeCount = size(tlTree_[d]);
    tlSplits_.resize(nodeCount);
    for (Split& split : tlSplits_)
        split.isInit = false;
}

//......................................................................................................................

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initOrderedSamples_(size_t usedVariableIndex, size_t usedSampleCount) const
{
    tlSampleBuffer_.resize(usedSampleCount);
    std::span sortedUsedSamples{ tlSampleBuffer_ };

    size_t variableIndex = tlUsedVariables_[usedVariableIndex];
    std::span sortedSamples{ sortedSamples_[variableIndex] };

    std::span sampleMask{ tlSampleStatus_ };

    auto p = cbegin(sortedSamples);
    auto q = begin(sortedUsedSamples);
    while (q != end(sortedUsedSamples)) {
        SampleIndex i = *p;
        *q = i;
        ++p;
        q += sampleMask[i];
    }

    tlOrderedSampleIndex_ = { begin(sortedUsedSamples), end(sortedUsedSamples) };
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initOrderedSamplesAlt_(size_t usedVariableIndex, size_t usedSampleCount) const
{
    initOrderedSamples_(usedVariableIndex, usedSampleCount);
    swap(tlSampleBuffer_, tlOrderedSamplesByVariable_[usedVariableIndex]);
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateOrderedSamples_(size_t usedVariableIndex, size_t usedSampleCount) const
{
    // status = 0 means unused sample
    // status = k + 1 means sample belongs to child node k

    const size_t unusedSampleCount = sampleCount_ - usedSampleCount;
    std::span sampleCountByChildNode{ tlSampleCountByChildNode_ };
    size_t childNodeCount = size(sampleCountByChildNode);

    tlSampleBuffer_.resize(sampleCount_);
    auto p = cbegin(std::span(tlSampleBuffer_));

    tlOrderedSampleIndex_.resize(childNodeCount + 1);
    auto q = begin(tlOrderedSampleIndex_);

    for (size_t k = 0; k < childNodeCount + 1; ++k) {
        q[k] = p;
        p += (k == 0 ? unusedSampleCount : sampleCountByChildNode[k - 1]);
    }

    auto s = cbegin(tlSampleStatus_);

    size_t variableIndex = tlUsedVariables_[usedVariableIndex];
    std::span sortedSamples{ sortedSamples_[variableIndex] };
    for (SampleIndex i : sortedSamples)
        *(q[s[i]]++) = i;

    // no need to call initOrderedSampleIndex_()
    // tlOrderedSampleIndex_[k] now points to the end of the block with status k samples
    // which is the beginning of the block with child node k samples
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateOrderedSamplesAlt_(size_t usedVariableIndex, size_t usedSampleCount, size_t d) const
{
    tlSampleBuffer_.resize(usedSampleCount);

    std::span parentNodes{ tlTree_[d - 1] };
    std::span sampleCountByParentNode{ tlSampleCountByParentNode_ };
    std::span sampleCountByChildNode{ tlSampleCountByChildNode_ };
    std::span prevOrderedUsedSamples{ tlOrderedSamplesByVariable_[usedVariableIndex] };
    std::span orderedUsedSamples{ tlSampleBuffer_ };

    const size_t parentNodeCount = size(sampleCountByParentNode);

    auto p = cbegin(prevOrderedUsedSamples);

    initOrderedSampleIndex_(orderedUsedSamples);
    auto q = begin(tlOrderedSampleIndex_);

    auto s = cbegin(tlSampleStatus_);

    for (size_t parentNodeIndex = 0; parentNodeIndex < parentNodeCount; ++parentNodeIndex) {

        auto pEnd = p + sampleCountByParentNode[parentNodeIndex];

        const TreeNode* parentNode = &parentNodes[parentNodeIndex];
        if (!parentNode->isLeaf)
            for(SampleIndex i: std::span(p, pEnd))
                *(q[s[i] - 1]++) = i;
        
        p = pEnd;
    }

    ASSERT(p == cend(prevOrderedUsedSamples));

    initOrderedSampleIndex_(orderedUsedSamples);

    swap(tlOrderedSamplesByVariable_[usedVariableIndex], tlSampleBuffer_);
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initOrderedSampleIndex_(std::span<SampleIndex> orderedSamples) const
{
    std::span sampleCountByChildNode{ tlSampleCountByChildNode_ };
    const size_t childNodeCount = size(sampleCountByChildNode);

    tlOrderedSampleIndex_.resize(childNodeCount + 1);
    std::span orderedSampleIndex{ tlOrderedSampleIndex_ };

    auto q = begin(orderedSamples);
    for (size_t childNodeIndex = 0; childNodeIndex < childNodeCount; ++childNodeIndex) {
        orderedSampleIndex[childNodeIndex] = q;
        q += sampleCountByChildNode[childNodeIndex];
    }
    orderedSampleIndex[childNodeCount] = q;
    ASSERT(q == end(orderedSamples));
}

//......................................................................................................................

template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateSplits_(
    size_t usedVariableIndex,
    CRefXd outData,
    CRefXd weights,
    const TreeOptions& options
) const
{
    size_t variableIndex = tlUsedVariables_[usedVariableIndex];
    std::span orderedSampleIndex{ tlOrderedSampleIndex_ };
    std::span<typename TreeTrainerImpl<SampleIndex>::Split> splits{ tlSplits_ };
    for (size_t k = 0; k != size(splits); ++k) {
        std::span sortedSamples{ orderedSampleIndex[k], orderedSampleIndex[k + 1] };
        updateSplit_(&splits[k], variableIndex, sortedSamples, outData, weights, options);
    }
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateSplit_(
    typename TreeTrainerImpl<SampleIndex>::Split* split,
    size_t j,
    std::span<SampleIndex> sortedSamples,
    CRefXd outData,
    CRefXd weights,
    const TreeOptions& options) const
{
    const float* pInDataColJ = &inData_.coeffRef(0, j);
    const double* pOutData = &outData.coeffRef(0);
    const double* pWeights = &weights.coeffRef(0);

    if (!split->isInit) {
        PROFILE::PUSH(PROFILE::SUMS);
        size_t ITEM_COUNT = size(sortedSamples);
            
        split->splitFound = false;
        split->iterationCount = 0;
        split->slowBranchCount = 0;

        double sumW = 0.0;
        double sumWY = 0.0;
        for (SampleIndex i: sortedSamples) {
            double w = pWeights[i];
            double y = pOutData[i];
            sumW += w;
            sumWY += w * y;
        }
        split->sumW = sumW;
        split->sumWY = sumWY;
        split->score = sumWY * sumWY / sumW + options.minGain();
        double tol = sumW * sqrt(static_cast<double>(size(sortedSamples))) * numeric_limits<double>::epsilon() / 2;
        split->minNodeWeight = std::max<double>(options.minNodeWeight(), tol);

        split->isInit = true;

        PROFILE::POP(ITEM_COUNT);
    }

    if (split->sumW == 0) return;

    const size_t minNodeSize = options.minNodeSize();
    const double minNodeWeight = split->minNodeWeight;
    size_t slowBranchCount = 0;

    bool splitFound = split->splitFound;
    double bestScore = split->score;
    size_t bestJ = split->j;
    float bestX = split->x;
    double bestLeftY = split->leftY;
    double bestRightY = split->rightY;
    size_t bestLeftSampleCount = split->leftSampleCount;

    double leftSumW = 0.0;
    double leftSumWY = 0.0;
    double rightSumW = split->sumW;
    double rightSumWY = split->sumWY;

    auto p = begin(sortedSamples);
    size_t nextI = *p;
    while (p != end(sortedSamples) - 1) {

        // this is where most execution time is spent
        const size_t i = nextI;
        nextI = *++p;
        const double w = pWeights[i];
        const double y = pOutData[i];
        leftSumW += w;
        rightSumW -= w;
        leftSumWY += w * y;
        rightSumWY -= w * y;
        const double score = leftSumWY * leftSumWY / leftSumW + rightSumWY * rightSumWY / rightSumW;

        if (score <= bestScore) continue;  // usually true

        ++slowBranchCount;

        if (p < begin(sortedSamples) + minNodeSize
            || p > end(sortedSamples) - minNodeSize
            || leftSumW < minNodeWeight
            || rightSumW < minNodeWeight
            ) continue;

        const float leftX = pInDataColJ[i];
        const float rightX = pInDataColJ[nextI];
        const float midX = (leftX + rightX) / 2;

        if (leftX == midX) continue;

        bestScore = score;
        bestJ = j;
        bestX = midX;
        bestLeftY = leftSumWY / leftSumW;
        bestRightY = rightSumWY / rightSumW;
        bestLeftSampleCount = p - begin(sortedSamples);
        splitFound = true;
    }

    split->splitFound = splitFound;
    split->score = bestScore;
    split->j = bestJ;
    split->x = bestX;
    split->leftY = bestLeftY;
    split->rightY = bestRightY;
    split->leftSampleCount = bestLeftSampleCount;
    split->rightSampleCount = size(sortedSamples) - bestLeftSampleCount;

    split->iterationCount += size(sortedSamples);
    split->slowBranchCount += slowBranchCount;
}

//......................................................................................................................

template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::updateTree_(size_t d) const
{
    std::span parentNodes{ tlTree_[d] };
    vector<TreeNode>& childNodes{ tlTree_[d + 1] };
    std::span splits{ tlSplits_ };
    std::swap(tlSampleCountByChildNode_, tlSampleCountByParentNode_);
    vector<size_t>& sampleCountByChildNode{ tlSampleCountByChildNode_ };

    size_t parentCount = size(parentNodes);
    size_t maxChildCount = 2 * parentCount;
    childNodes.resize(maxChildCount);
    sampleCountByChildNode.resize(maxChildCount);

    TreeNode* pParentNode = data(parentNodes);
    TreeNode* pChildNode = data(childNodes);
    size_t* pSampleCountByChildNode = data(sampleCountByChildNode);
    Split* pSplit = data(splits);


    for (size_t k = 0; k < parentCount; ++k) {

        double sumW = pSplit->sumW;
        double sumWY = pSplit->sumWY;
        double y = (sumW == 0) ? 0.0 : sumWY / sumW;

        pParentNode->y = static_cast<float>(y);

        if (pSplit->splitFound) {

            pParentNode->isLeaf = false;
            pParentNode->j = static_cast<uint32_t>(pSplit->j);
            pParentNode->x = pSplit->x;
            pParentNode->gain = static_cast<float>(pSplit->score - sumWY * sumWY / sumW);

            pParentNode->leftChild = pChildNode;
            pChildNode->isLeaf = true;
            pChildNode->y = static_cast<float>(pSplit->leftY);
            ++pChildNode;
            *pSampleCountByChildNode = pSplit->leftSampleCount;
            ++pSampleCountByChildNode;

            pParentNode->rightChild = pChildNode;
            pChildNode->isLeaf = true;
            pChildNode->y = static_cast<float>(pSplit->rightY);
            ++pChildNode;
            *pSampleCountByChildNode = pSplit->rightSampleCount;
            ++pSampleCountByChildNode;
        }
        else
            pParentNode->isLeaf = true;
        ++pParentNode;
        ++pSplit;
    }
    size_t childCount = pChildNode - childNodes.data();
    childNodes.resize(childCount);
    sampleCountByChildNode.resize(childCount);

    size_t usedSampleCount = std::accumulate(begin(sampleCountByChildNode), end(sampleCountByChildNode), 0LL);
    return usedSampleCount;
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateSampleStatus_(size_t d) const
{
    std::span parentNodes{ tlTree_[d - 1] };
    std::span childNodes{ tlTree_[d] };
    std::span sampleStatus{ tlSampleStatus_ };

    for (size_t i = 0; i < sampleCount_; ++i) {
        if (sampleStatus[i] == 0) continue;
        SampleIndex s = sampleStatus[i];
        const TreeNode* node = &parentNodes[s - 1];
        if (node->isLeaf) {
            sampleStatus[i] = 0;
            continue;
        }
        node = (inData_(i, node->j) < node->x) ? node->leftChild : node->rightChild;
        s = static_cast<SampleIndex>(node - (&childNodes[0] - 1));
        sampleStatus[i] = s;
    }
}

//......................................................................................................................

template<typename SampleIndex>
unique_ptr<BasePredictor> TreeTrainerImpl<SampleIndex>::createPredictor_() const
{
    const TreeNode* root = data(tlTree_.front());

    size_t treeDepth = depth(root);
    if (treeDepth == 0)
        return std::make_unique<TrivialPredictor>(root->y);
    if (treeDepth == 1)
        return std::make_unique<StumpPredictor>(root->j, root->x, root->leftChild->y, root->rightChild->y, root->gain);

    auto [root1, nodes1] = cloneBreadthFirst(root);     // depth or breadth does not matter
    return std::make_unique<TreePredictor>(root1, move(nodes1));
}

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImpl<uint8_t>;
template class TreeTrainerImpl<uint16_t>;
template class TreeTrainerImpl<uint32_t>;
template class TreeTrainerImpl<uint64_t>;

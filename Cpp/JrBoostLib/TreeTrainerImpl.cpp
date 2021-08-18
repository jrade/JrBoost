//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImpl.h"
#include "TreeOptions.h"
#include "StumpPredictor.h"
#include "TrivialPredictor.h"
#include "TreePredictor.h"
#include "pdqsort.h"

#ifdef _MSC_VER
    #pragma warning(disable: 4701)
#endif

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
    // profiling zero calibration
    PROFILE::PUSH(PROFILE::ZERO);
    size_t ITEM_COUNT = 1;

    // validate data
    //PROFILE::SWITCH(ITEM_COUNT, PROFILE::VALIDATE);
    //validateData_(outData, weights);
    //ITEM_COUNT = sampleCount_;

    // init used variables
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::USED_VARIABLES);
    ITEM_COUNT = initUsedVariables_(options);

    // init root
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_TREE);
    ITEM_COUNT = 0;
    nodes_.resize(std::max(options.maxDepth() + 1, size(nodes_)));
    vector<TreeNode>* pNodes = data(nodes_);
    pNodes[0].resize(1);

   // init sample status
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SAMPLE_STATUS);
    ITEM_COUNT = sampleCount_;
    initSampleStatus_(options, weights);
    const size_t* pSampleCountByStatus = data(sampleCountByStatus_);

    size_t d = 0;
    while (true) {

        size_t parentCount = size(pNodes[d]);

        splitData_.resize(parentCount);
        SplitData* pSplitData = data(splitData_);
        for (size_t k = 0; k < parentCount; ++k)
            pSplitData[k].isInit = false;

        for (size_t j : usedVariables_) {

            // init ordered samples
            PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES_C);
            ITEM_COUNT = sampleCount_;
            initOrderedSamples_(j);

            // find best split
            PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_NODE_BUILDER);
            ITEM_COUNT = 0;
            const SampleIndex* pSamples = data(sampleBuffer_) + pSampleCountByStatus[0];
            for (size_t k = 0; k != parentCount; ++k) {
                size_t n = pSampleCountByStatus[k + 1];
                updateSplit_(&pSplitData[k], outData, weights, options, pSamples, n, j);
                pSamples += n;
                ITEM_COUNT += n;
            }
        }

        if (omp_get_thread_num() == 0) {
            for (size_t k = 0; k != parentCount; ++k) {
                PROFILE::SPLIT_ITERATION_COUNT += pSplitData[k].iterationCount;
                PROFILE::SLOW_BRANCH_COUNT += pSplitData[k].slowBranchCount;
            }
        }

        // update tree
        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_TREE);
        ITEM_COUNT = 0;
        updateTree_(pNodes[d], pNodes[d + 1]);

        d += 1;
        if (d == options.maxDepth() || pNodes[d].empty())
            break;

        // update sample status
        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SAMPLE_STATUS);
        ITEM_COUNT = sampleCount_;
        updateSampleStatus_(pNodes[d - 1], pNodes[d]);
        pSampleCountByStatus = data(sampleCountByStatus_);
    }

    PROFILE::POP(ITEM_COUNT);

    // DO PRUNING HERE

    const TreeNode* root = pNodes[0].data();

    size_t treeDepth = depth(root);
    if (treeDepth == 0)
        return std::make_unique<TrivialPredictor>(root->y);
    if (treeDepth == 1)
        return std::make_unique<StumpPredictor>(root->j, root->x, root->leftChild->y, root->rightChild->y, root->gain);

    auto [root1, nodes1] = cloneBreadthFirst(root);     // depth or breadth does not matter
    return std::make_unique<TreePredictor>(root1, move(nodes1));
};


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
size_t TreeTrainerImpl<SampleIndex>::initUsedVariables_(const TreeOptions& options) const
{
    RandomNumberEngine_& rne = rne_;
    size_t candidateVariableCount = std::min(variableCount_, options.topVariableCount());
    size_t usedVariableCount = static_cast<size_t>(options.usedVariableRatio() * candidateVariableCount + 0.5);
    if (usedVariableCount == 0) usedVariableCount = 1;

    ASSERT(0 < usedVariableCount);
    ASSERT(usedVariableCount <= candidateVariableCount);
    ASSERT(candidateVariableCount <= variableCount_);

    usedVariables_.resize(usedVariableCount);
    size_t n = candidateVariableCount;
    size_t k = usedVariableCount;
    size_t i = 0;
    auto p = begin(usedVariables_);
    while (k > 0) {
        *p = i;
        bool b = BernoulliDistribution(k, n)(rne);
        p += b;
        k -= b;
        --n;
        ++i;
    }

    return candidateVariableCount;
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initSampleStatus_(const TreeOptions& options, CRefXd weights) const
{
    RandomNumberEngine_& rne = rne_;
    const size_t minNodeSize = options.minNodeSize();

    size_t usedSampleCount;
    sampleStatus_.resize(sampleCount_);

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

            for (auto p = begin(sampleStatus_); p != end(sampleStatus_); ++p) {
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
            for (auto p = begin(sampleStatus_); p != end(sampleStatus_); ++p) {
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

        sampleBuffer_.resize(sampleCount_);
        sampleStatus_.assign(sampleCount_, 0);
        SampleIndex* pSampleStatus = data(sampleStatus_);

        if (!options.isStratified()) {
            auto p = begin(sampleBuffer_);
            for (size_t i = 0; i < sampleCount_; ++i) {
                bool b = weights(i) >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
            }
            sampleBuffer_.resize(p - begin(sampleBuffer_));

            size_t n = size(sampleBuffer_);
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0 && n > 0) k = 1;
            usedSampleCount = k;

            for (SampleIndex i : sampleBuffer_) {
                bool b = BernoulliDistribution(k, n) (rne);
                pSampleStatus[i] = b;
                k -= b;
                --n;
            }
        }

        else {
            array<size_t, 2> n = { 0, 0 };
            auto p = begin(sampleBuffer_);
            for (size_t i = 0; i < sampleCount_; ++i) {
                size_t stratum = strata_[i];
                bool b = weights(i) >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
                n[stratum] += b;
            }
            sampleBuffer_.resize(p - begin(sampleBuffer_));

            array<size_t, 2> k;
            k[0] = static_cast<size_t>(options.usedSampleRatio() * n[0] + 0.5);
            if (k[0] == 0 && n[0] > 0) k[0] = 1;
            k[1] = static_cast<size_t>(options.usedSampleRatio() * n[1] + 0.5);
            if (k[1] == 0 && n[1] > 0) k[1] = 1;
            usedSampleCount = k[0] + k[1];

            for (size_t i : sampleBuffer_) {
                size_t stratum = strata_[i];
                bool b = BernoulliDistribution(k[stratum], n[stratum]) (rne);
                pSampleStatus[i] = b;
                k[stratum] -= b;
                --n[stratum];
            }
        }
    }

    sampleCountByStatus_ = { sampleCount_ - usedSampleCount, usedSampleCount };
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateSampleStatus_(
    const vector<TreeNode>& parentNodes, const vector<TreeNode>& childNodes) const
{
    SampleIndex* pSampleStatus = data(sampleStatus_);
    size_t statusCount = size(childNodes) + 1;
    sampleCountByStatus_.assign(statusCount, 0);
    for (size_t i = 0; i < sampleCount_; ++i) {
        if (pSampleStatus[i] == 0) continue;
        SampleIndex status = pSampleStatus[i];
        const TreeNode* node = &parentNodes[status - 1];
        if (node->isLeaf) {
            pSampleStatus[i] = 0;
            continue;
        }
        node = (inData_(i, node->j) < node->x) ? node->leftChild : node->rightChild;
        status = static_cast<SampleIndex>(node - (&childNodes[0] - 1));
        pSampleStatus[i] = status;
        ++sampleCountByStatus_[status];
    }
    sampleCountByStatus_[0] = sampleCount_ - std::accumulate(cbegin(sampleCountByStatus_) + 1, cend(sampleCountByStatus_), 0LL);
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initOrderedSamples_(size_t j) const
{
    size_t statusCount = size(sampleCountByStatus_);
    ASSERT(statusCount >= 2);
    if (statusCount == 2)
        return initOrderedSamplesFast_(j);

    vector<SampleIndex*> pSortedSamplesByStatus(statusCount);
    sampleBuffer_.resize(sampleCount_);
    SampleIndex* p = data(sampleBuffer_);
    for (size_t k = 0; k < statusCount; ++k) {
        pSortedSamplesByStatus[k] = p;
        p += sampleCountByStatus_[k];
    }

    auto pSortedSamples = cbegin(sortedSamples_[j]);
    auto pSortedSamplesEnd = cend(sortedSamples_[j]);
    auto l = begin(pSortedSamplesByStatus);
    auto pSampleStatus = cbegin(sampleStatus_);
    while (pSortedSamples != pSortedSamplesEnd) {
        SampleIndex i = *(pSortedSamples++);
        *(pSortedSamplesByStatus[pSampleStatus[i]]++) = i;
    }
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::initOrderedSamplesFast_(size_t j) const
{
    sampleBuffer_.resize(sampleCount_);
    const SampleIndex* p = data(sortedSamples_[j]);

    SampleIndex* q = data(sampleBuffer_) + sampleCountByStatus_[0];
    SampleIndex* q_end = q + sampleCountByStatus_[1];
    ASSERT(q_end == data(sampleBuffer_) + sampleCount_);

    const SampleIndex* pSampleStatus = data(sampleStatus_);

    while (q != q_end) {
        SampleIndex i = *p;
        *q = i;
        ++p;
        q += pSampleStatus[i];
    }
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateSplit_(
    typename TreeTrainerImpl<SampleIndex>::SplitData* splitData,
    CRefXd outData, CRefXd weights, const TreeOptions& options, 
    const SampleIndex* usedSamples, size_t usedSampleCount, size_t j) const
{
    const SampleIndex* pBegin = usedSamples;
    const SampleIndex* pEnd = usedSamples + usedSampleCount;

    const float* pInDataColJ = &inData_.coeffRef(0, j);
    const double* pOutData = &outData.coeffRef(0);
    const double* pWeights = &weights.coeffRef(0);

    if (!splitData->isInit) {
        PROFILE::PUSH(PROFILE::SUMS);
        size_t ITEM_COUNT = usedSampleCount;
            
        splitData->splitFound = false;
        splitData->iterationCount = 0;
        splitData->slowBranchCount = 0;

        double sumW = 0.0;
        double sumWY = 0.0;
        for (const SampleIndex* p = pBegin; p != pEnd; ++p) {
            size_t i = *p;
            double w = pWeights[i];
            double y = pOutData[i];
            sumW += w;
            sumWY += w * y;
        }
        splitData->sumW = sumW;
        splitData->sumWY = sumWY;
        splitData->score = sumWY * sumWY / sumW + options.minGain();
        double tol = sumW * sqrt(static_cast<double>(usedSampleCount)) * numeric_limits<double>::epsilon() / 2;
        splitData->minNodeWeight = std::max<double>(options.minNodeWeight(), tol);

        splitData->isInit = true;

        PROFILE::POP(ITEM_COUNT);
    }

    if (splitData->sumW == 0) return;

    const size_t minNodeSize = options.minNodeSize();
    const double minNodeWeight = splitData->minNodeWeight;
    size_t slowBranchCount = 0;

    bool splitFound = splitData->splitFound;
    double bestScore = splitData->score;
    size_t bestJ = splitData->j;
    float bestX = splitData->x;
    double bestLeftY = splitData->leftY;
    double bestRightY = splitData->rightY;

    double leftSumW = 0.0;
    double leftSumWY = 0.0;
    double rightSumW = splitData->sumW;
    double rightSumWY = splitData->sumWY;

    const SampleIndex* p = pBegin;
    size_t nextI = *p;
    while (p != pEnd - 1) {

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

        if (p < pBegin + minNodeSize
            || p > pEnd - minNodeSize
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
        splitFound = true;
    }

    splitData->splitFound = splitFound;
    splitData->score = bestScore;
    splitData->j = bestJ;
    splitData->x = bestX;
    splitData->leftY = bestLeftY;
    splitData->rightY = bestRightY;

    splitData->iterationCount += usedSampleCount;
    splitData->slowBranchCount += slowBranchCount;
}


template<typename SampleIndex>
void TreeTrainerImpl<SampleIndex>::updateTree_(vector<TreeNode>& parentNodes, vector<TreeNode>& childNodes) const
{
    const SplitData* pSplitData = data(splitData_);
    size_t parentCount = size(parentNodes);
    size_t maxChildCount = 2 * parentCount;
    childNodes.resize(maxChildCount);

    TreeNode* pParent = parentNodes.data();
    TreeNode* pChild = childNodes.data();
    for (size_t k = 0; k < parentCount; ++k) {

        double sumW = pSplitData[k].sumW;
        double sumWY = pSplitData[k].sumWY;
        double y = (sumW == 0) ? 0.0 : sumWY / sumW;
        pParent->y = static_cast<float>(y);

        if (pSplitData[k].splitFound) {

            pParent->isLeaf = false;
            pParent->j = static_cast<uint32_t>(pSplitData[k].j);
            pParent->x = pSplitData[k].x;
            pParent->gain = static_cast<float>(pSplitData[k].score - sumWY * sumWY / sumW);

            pParent->leftChild = pChild;
            pChild->isLeaf = true;
            pChild->y = static_cast<float>(pSplitData[k].leftY);
            ++pChild;

            pParent->rightChild = pChild;
            pChild->isLeaf = true;
            pChild->y = static_cast<float>(pSplitData[k].rightY);
            ++pChild;
        }
        else
            pParent->isLeaf = true;
        ++pParent;
    }
    size_t childCount = pChild - childNodes.data();
    childNodes.resize(childCount);
}

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImpl<uint8_t>;
template class TreeTrainerImpl<uint16_t>;
template class TreeTrainerImpl<uint32_t>;
template class TreeTrainerImpl<uint64_t>;

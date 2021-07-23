//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImplD.h"
#include "NodeBuilder.h"
#include "StumpPredictor.h"
#include "TreeOptions.h"
#include "TreePredictor.h"
#include "TrivialPredictor.h"
#include "pdqsort.h"


static size_t depth_(const TreeNode* node)
{
    if (node->isLeaf) return 0;
    return 1 + std::max(depth_(node->leftChild), depth_(node->rightChild));
}

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
TreeTrainerImplD<SampleIndex>::TreeTrainerImplD(CRefXXf inData, CRefXs strata) :
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
vector<vector<SampleIndex>> TreeTrainerImplD<SampleIndex>::createSortedSamples_() const
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
unique_ptr<BasePredictor> TreeTrainerImplD<SampleIndex>::train(
    CRefXd outData, CRefXd weights, const TreeOptions& options) const
{
    PROFILE::PUSH(PROFILE::ZERO);
    size_t ITEM_COUNT = 1;

    //PROFILE::SWITCH(ITEM_COUNT, PROFILE::VALIDATE);
    //validateData_(outData, weights);
    //ITEM_COUNT = sampleCount_;

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::USED_VARIABLES);
    ITEM_COUNT = initUsedVariables_(options);

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SAMPLE_MASK);
    size_t usedSampleCount = initSampleMask_(options, weights);
    vector<size_t> sampleCountByParentNode = { usedSampleCount };
    ITEM_COUNT = sampleCount_;

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES_D);
    orderedSamples_.resize(variableCount_);
    for (size_t j : usedVariables_)
        initOrderedSamples_(j, usedSampleCount);
    ITEM_COUNT = size(usedVariables_) * sampleCount_;

    PROFILE::POP(ITEM_COUNT);


    // create the root
    vector<vector<TreeNode>> nodes(options.maxDepth() + 1);
    nodes[0].resize(1);
    vector<NodeBuilder<SampleIndex>> nodeBuilders;

    for (size_t d = 0;; ++d) {

        size_t parentCount = size(nodes[d]);
        TreeNode* pParents = data(nodes[d]);

        // find best splits

        nodeBuilders.resize(parentCount);
        for (size_t k = 0; k < parentCount; ++k)
            nodeBuilders[k].reset(inData_, outData, weights, options);

        PROFILE::PUSH(PROFILE::UPDATE_NODE_BUILDER);
        ITEM_COUNT = 0;
        for (size_t j : usedVariables_) {
            ITEM_COUNT += size(orderedSamples_[j]);
            const SampleIndex* s = data(orderedSamples_[j]);
            for (size_t k = 0; k < parentCount; ++k) {
                nodeBuilders[k].update(j, s, s + sampleCountByParentNode[k]);
                s += sampleCountByParentNode[k];
            }
            ASSERT(s == data(orderedSamples_[j]) + size(orderedSamples_[j]));
        }
        PROFILE::POP(ITEM_COUNT);

        // create child nodes

        size_t maxChildCount = 2 * parentCount;
        nodes[d + 1].resize(maxChildCount);
        vector<size_t> sampleCountByChildNode(maxChildCount);
        TreeNode* pChildren = data(nodes[d + 1]);

        // update parent and init child nodes

        TreeNode* p = pParents;
        TreeNode* c = pChildren;
        size_t* n = data(sampleCountByChildNode);
        for (size_t k = 0; k < parentCount; ++k)
            nodeBuilders[k].initNodes(&p, &c, &n);
        size_t childCount = c - pChildren;
        nodes[d + 1].resize(childCount);
        sampleCountByChildNode.resize(childCount);

        if (d + 1 == options.maxDepth() || childCount == 0) break;

        // update ordered samples

        PROFILE::PUSH(PROFILE::UPDATE_ORDERED_SAMPLES);
        ITEM_COUNT = 0;
        for (size_t j : usedVariables_) {
            ITEM_COUNT += size(orderedSamples_[j]);
            updateOrderedSamples_(j, nodes[d], sampleCountByParentNode, sampleCountByChildNode);
        }
        PROFILE::POP(ITEM_COUNT);

        sampleCountByParentNode = move(sampleCountByChildNode);
    }

    if (omp_get_thread_num() == 0) {
        PROFILE::SPLIT_ITERATION_COUNT += NodeBuilder<SampleIndex>::iterationCount();
        PROFILE::SLOW_BRANCH_COUNT += NodeBuilder<SampleIndex>::slowBranchCount();
    }

    TreeNode* root = &nodes[0][0];

    if (options.pruneFactor() != 0.0)
        prune(root, static_cast<float>(options.pruneFactor() * root->gain));

    //size_t d = depth_(root);
    //if (d == 0)
    //    return std::make_unique<TrivialPredictor>(root->y);
    //else if (d == 1)
    //    return std::make_unique<StumpPredictor>(root->j, root->x, root->leftChild->y, root->rightChild->y);
    //else
        return std::make_unique<TreePredictor>(root, move(nodes));
};


template<typename SampleIndex>
void TreeTrainerImplD<SampleIndex>::validateData_(CRefXd outData, CRefXd weights) const
{
    if (static_cast<size_t>(outData.rows()) != sampleCount_)
        throw std::invalid_argument("Train indata and outdata have different numbers of samples.");
    if (!(outData.abs() < numeric_limits<double>::infinity()).all())
        throw std::invalid_argument("Train outdata has values that are infinity or NaN.");

    if (static_cast<size_t>(weights.rows()) != sampleCount_)
        throw std::invalid_argument("Train indata and weights have different numbers of samples.");
    if (!(weights.abs() < numeric_limits<float>::infinity()).all())
        throw std::invalid_argument("Train weights have values that are infinity or NaN.");
    if ((weights < 0.0).any())
        throw std::invalid_argument("Train weights have negative values.");
}


template<typename SampleIndex>
size_t TreeTrainerImplD<SampleIndex>::initUsedVariables_(const TreeOptions& options) const
{
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
        bool b = BernoulliDistribution(k, n)(rne_);
        p += b;
        k -= b;
        --n;
        ++i;
    }

    return candidateVariableCount;
}


template<typename SampleIndex>
size_t TreeTrainerImplD<SampleIndex>::initSampleMask_(const TreeOptions& options, CRefXd weights) const
{
    size_t usedSampleCount;

    double minSampleWeight = options.minAbsSampleWeight();
    double minRelSampleWeight = options.minRelSampleWeight();
    if (minRelSampleWeight > 0.0)
        minSampleWeight = std::max(minSampleWeight, weights.maxCoeff() * minRelSampleWeight);

    bool isStratified = options.isStratified();

    if (minSampleWeight == 0.0) {

        sampleMask_.resize(sampleCount_);

        if (!isStratified) {

            // create a random mask of length n with k ones and n - k zeros
            // n = total number of samples
            // k = number of used samples

            size_t n = sampleCount_;
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0) k = 1;
            usedSampleCount = k;

            for (auto p = begin(sampleMask_); p != end(sampleMask_); ++p) {
                bool b = BernoulliDistribution(k, n) (rne_);
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
            for (auto p = begin(sampleMask_); p != end(sampleMask_); ++p) {
                size_t stratum = *s;
                ++s;
                bool b = BernoulliDistribution(k[stratum], n[stratum]) (rne_);
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

        tmpSamples_.resize(sampleCount_);
        sampleMask_.assign(sampleCount_, 0);

        if (!isStratified) {
            auto p = begin(tmpSamples_);
            for (size_t i = 0; i < sampleCount_; ++i) {
                bool b = weights(i) >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
            }
            tmpSamples_.resize(p - begin(tmpSamples_));

            size_t n = size(tmpSamples_);
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0 && n > 0) k = 1;
            usedSampleCount = k;

            for (SampleIndex i : tmpSamples_) {
                bool b = BernoulliDistribution(k, n) (rne_);
                sampleMask_[i] = b;
                k -= b;
                --n;
            }
        }

        else {
            array<size_t, 2> n = { 0, 0 };
            auto p = begin(tmpSamples_);
            for (size_t i = 0; i < sampleCount_; ++i) {
                size_t stratum = strata_[i];
                bool b = weights(i) >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
                n[stratum] += b;
            }
            tmpSamples_.resize(p - begin(tmpSamples_));

            array<size_t, 2> k;
            k[0] = static_cast<size_t>(options.usedSampleRatio() * n[0] + 0.5);
            if (k[0] == 0 && n[0] > 0) k[0] = 1;
            k[1] = static_cast<size_t>(options.usedSampleRatio() * n[1] + 0.5);
            if (k[1] == 0 && n[1] > 0) k[1] = 1;
            usedSampleCount = k[0] + k[1];

            for (size_t i : tmpSamples_) {
                size_t stratum = strata_[i];
                bool b = BernoulliDistribution(k[stratum], n[stratum]) (rne_);
                sampleMask_[i] = b;
                k[stratum] -= b;
                --n[stratum];
            }
        }
    }

    return usedSampleCount;
}


template<typename SampleIndex>
void TreeTrainerImplD<SampleIndex>::initOrderedSamples_(size_t j, size_t usedSampleCount) const
{
    const SampleIndex* p = data(sortedSamples_[j]);

    orderedSamples_[j].resize(usedSampleCount);
    SampleIndex* q = data(orderedSamples_[j]);
    SampleIndex* q_end = q + usedSampleCount;

    const char* pSampleMask = data(sampleMask_);

    while (q != q_end) {
        SampleIndex i = *p;
        *q = i;
        ++p;
        q += pSampleMask[i];
    }
}


template<typename SampleIndex>
void TreeTrainerImplD<SampleIndex>::updateOrderedSamples_(
    size_t j,
    const vector<TreeNode>& nodes,
    const vector<size_t>& sampleCountByParentNode,
    const vector<size_t>& sampleCountByChildNode
) const
{
    tmpSamples_.resize(size(orderedSamples_[j]));

    const SampleIndex* p = data(orderedSamples_[j]);
    SampleIndex* leftQ = data(tmpSamples_);

    size_t parentNodeCount = size(sampleCountByParentNode);
    size_t childNodeCount = size(sampleCountByChildNode);

    size_t childNodeIndex = 0;
    for (size_t parentNodeIndex = 0; parentNodeIndex < parentNodeCount; ++parentNodeIndex) {

        const TreeNode* parentNode = &nodes[parentNodeIndex];

        if (parentNode->isLeaf) {
            p += sampleCountByParentNode[parentNodeIndex];
            continue;
        }

        size_t j1 = parentNode->j;
        float x = parentNode->x;
        SampleIndex* rightQ = leftQ + sampleCountByChildNode[childNodeIndex];
        const SampleIndex* pEnd = p + sampleCountByParentNode[parentNodeIndex];
        while (p != pEnd) {
            SampleIndex i = *(p++);
            *((inData_(i, j1) < x ? leftQ : rightQ)++) = i;
        }
        ASSERT(rightQ == leftQ + sampleCountByChildNode[childNodeIndex + 1]);
        leftQ = rightQ;
        childNodeIndex += 2;
    }
    ASSERT(childNodeIndex == childNodeCount);

    tmpSamples_.resize(leftQ - data(tmpSamples_));
    swap(orderedSamples_[j], tmpSamples_);
}

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImplD<uint8_t>;
template class TreeTrainerImplD<uint16_t>;
template class TreeTrainerImplD<uint32_t>;
template class TreeTrainerImplD<uint64_t>;

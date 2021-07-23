//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImplC.h"
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
TreeTrainerImplC<SampleIndex>::TreeTrainerImplC(CRefXXf inData, CRefXs strata) :
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
vector<vector<SampleIndex>> TreeTrainerImplC<SampleIndex>::createSortedSamples_() const
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
unique_ptr<BasePredictor> TreeTrainerImplC<SampleIndex>::train(
    CRefXd outData, CRefXd weights, const TreeOptions& options) const
{
    PROFILE::PUSH(PROFILE::ZERO);
    size_t ITEM_COUNT = 1;

    //PROFILE::SWITCH(ITEM_COUNT, PROFILE::VALIDATE);
    //validateData_(outData, weights);
    //ITEM_COUNT = sampleCount_;

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::USED_VARIABLES);
    ITEM_COUNT = initUsedVariables_(options);

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SAMPLE_STATUS);
    initSampleStatus_(options, weights);
    ITEM_COUNT = sampleCount_;

    PROFILE::POP(ITEM_COUNT);


    // create the root

    vector<vector<TreeNode>> nodes(options.maxDepth() + 1);
    nodes[0].resize(1);


    // split the root

    vector<NodeBuilder<SampleIndex>> nodeBuilders;
    nodeBuilders.resize(1);
    nodeBuilders[0].reset(inData_, outData, weights, options);
    usedSortedSamples_.resize(variableCount_);

    for (size_t j : usedVariables_) {

        PROFILE::PUSH(PROFILE::INIT_USED_SORTED_SAMPLES);
        ITEM_COUNT = sampleCount_;
        initUsedSortedSamples_(j);
        
        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_NODE_BUILDER);
        ITEM_COUNT = size(usedSortedSamples_[j]);
        nodeBuilders[0].update(
            j,
            data(usedSortedSamples_[j]),
            data(usedSortedSamples_[j]) + size(usedSortedSamples_[j])
        );

        PROFILE::POP(ITEM_COUNT);
    }

    size_t d = 0;
    size_t parentCount = 1;

    while(true) {

        // update the parent nodes and initialize the child nodes

        size_t maxChildCount = 2 * parentCount;
        nodes[d + 1].resize(maxChildCount);
        TreeNode* pParents = data(nodes[d]);
        TreeNode* pChildren = data(nodes[d + 1]);
        vector<size_t> sampleCountByNode(maxChildCount);

        TreeNode* p = pParents;
        TreeNode* c = pChildren;
        size_t* n = data(sampleCountByNode);
        for (size_t k = 0; k < parentCount; ++k)
            nodeBuilders[k].initNodes(&p, &c, &n);
        size_t childCount = c - pChildren;
        nodes[d + 1].resize(childCount);
        sampleCountByNode.resize(childCount);

        ++d;
        if (d == options.maxDepth() || childCount == 0) break;

        // update sample status

        PROFILE::PUSH(PROFILE::UPDATE_SAMPLE_STATUS);
        ITEM_COUNT = sampleCount_;
        updateSampleStatus_(pParents, pChildren);
        PROFILE::POP(ITEM_COUNT);

        // here the nodes that previously were children become parents

        parentCount = childCount;
        childCount = 0;

        nodeBuilders.resize(parentCount);
        for (size_t k = 0; k < parentCount; ++k)
            nodeBuilders[k].reset(inData_, outData, weights, options);

        for (size_t j : usedVariables_) {

            PROFILE::PUSH(PROFILE::UPDATE_USED_SORTED_SAMPLES);
            ITEM_COUNT = size(usedSortedSamples_[j]);
            updateUsedSortedSamples_(j);

            PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES_C);
            ITEM_COUNT = size(usedSortedSamples_[j]);
            initOrderedSamples_(j, sampleCountByNode);

            PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_NODE_BUILDER);
            ITEM_COUNT = size(orderedSamples_);
            const SampleIndex* s = data(orderedSamples_);
            for (size_t k = 0; k < parentCount; ++k) {
                nodeBuilders[k].update(
                    j,
                    s,
                    s + sampleCountByNode[k]);
                s += sampleCountByNode[k];
            }
            ASSERT(s == data(orderedSamples_) + size(orderedSamples_));

            PROFILE::POP(ITEM_COUNT);
        }
    }

    nodes.resize(d + 1);

    //if (omp_get_thread_num() == 0)
    //    std::cout << std::endl;

    if (omp_get_thread_num() == 0) {
        PROFILE::SPLIT_ITERATION_COUNT += NodeBuilder<SampleIndex>::iterationCount();
        PROFILE::SLOW_BRANCH_COUNT += NodeBuilder<SampleIndex>::slowBranchCount();
    }

    TreeNode* root = &nodes[0][0];

    if (options.pruneFactor() != 0.0)
        prune(root, static_cast<float>(options.pruneFactor() * root->gain));

    //d = depth_(root);
    //if (d == 0)
    //    return std::make_unique<TrivialPredictor>(root->y);
    //else if (d == 1)
    //    return std::make_unique<StumpPredictor>(root->j, root->x, root->leftChild->y, root->rightChild->y);
    //else
        return std::make_unique<TreePredictor>(root, move(nodes));
};


template<typename SampleIndex>
void TreeTrainerImplC<SampleIndex>::validateData_(CRefXd outData, CRefXd weights) const
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
size_t TreeTrainerImplC<SampleIndex>::initUsedVariables_(const TreeOptions& options) const
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
void TreeTrainerImplC<SampleIndex>::initSampleStatus_(const TreeOptions& options, CRefXd weights) const
{
    size_t usedSampleCount;

    double minSampleWeight = options.minAbsSampleWeight();
    double minRelSampleWeight = options.minRelSampleWeight();
    if (minRelSampleWeight > 0.0)
        minSampleWeight = std::max(minSampleWeight, weights.maxCoeff() * minRelSampleWeight);

    bool isStratified = options.isStratified();

    if (minSampleWeight == 0.0) {

        sampleStatus_.resize(sampleCount_);

        if (!isStratified) {

            // create a random mask of length n with k ones and n - k zeros
            // n = total number of samples
            // k = number of used samples

            size_t n = sampleCount_;
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0) k = 1;
            usedSampleCount = k;

            for (auto p = begin(sampleStatus_); p != end(sampleStatus_); ++p) {
                bool b = BernoulliDistribution(k, n) (rne_);
                *p = b ? 0 : UNUSED;
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
                bool b = BernoulliDistribution(k[stratum], n[stratum]) (rne_);
                *p = b ? 0 : UNUSED;
                k[stratum] -= b;
                --n[stratum];
            }
        }
    }

    else {  // minSampleWeight > 0.0

        // same as above, but first we identify the smaples with weight >= sampleMinWeight
        // these samples are stored in tmpSamples
        // then we select among those samples

        vector<SampleIndex> tmpSamples(sampleCount_);
        sampleStatus_.assign(sampleCount_, UNUSED);
        SampleIndex* pSampleStatus = data(sampleStatus_);

        if (!isStratified) {
            auto p = begin(tmpSamples);
            for (size_t i = 0; i < sampleCount_; ++i) {
                bool b = weights(i) >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
            }
            tmpSamples.resize(p - begin(tmpSamples));

            size_t n = size(tmpSamples);
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0 && n > 0) k = 1;
            usedSampleCount = k;

            for (SampleIndex i : tmpSamples) {
                bool b = BernoulliDistribution(k, n) (rne_);
                pSampleStatus[i] = b ? 0 : UNUSED;
                k -= b;
                --n;
            }
        }

        else {
            array<size_t, 2> n = { 0, 0 };
            auto p = begin(tmpSamples);
            for (size_t i = 0; i < sampleCount_; ++i) {
                size_t stratum = strata_[i];
                bool b = weights(i) >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
                n[stratum] += b;
            }
            tmpSamples.resize(p - begin(tmpSamples));

            array<size_t, 2> k;
            k[0] = static_cast<size_t>(options.usedSampleRatio() * n[0] + 0.5);
            if (k[0] == 0 && n[0] > 0) k[0] = 1;
            k[1] = static_cast<size_t>(options.usedSampleRatio() * n[1] + 0.5);
            if (k[1] == 0 && n[1] > 0) k[1] = 1;
            usedSampleCount = k[0] + k[1];

            for (size_t i : tmpSamples) {
                size_t stratum = strata_[i];
                bool b = BernoulliDistribution(k[stratum], n[stratum]) (rne_);
                pSampleStatus[i] = b ? 0 : UNUSED;
                k[stratum] -= b;
                --n[stratum];
            }
        }
    }
}


template<typename SampleIndex>
void TreeTrainerImplC<SampleIndex>::initUsedSortedSamples_(size_t j) const
{
    usedSortedSamples_[j].resize(size(sortedSamples_[j]));

    const SampleIndex* p = data(sortedSamples_[j]);
    const SampleIndex* p_end = data(sortedSamples_[j]) + size(sortedSamples_[j]);
    SampleIndex* q = data(usedSortedSamples_[j]);

    const SampleIndex* pSampleStatus = data(sampleStatus_);

    while (p != p_end) {
        SampleIndex i = *p;
        *q = i;
        ++p;
        q += (pSampleStatus[i] != UNUSED);
    }

    usedSortedSamples_[j].resize(q - data(usedSortedSamples_[j]));
}


template<typename SampleIndex>
void TreeTrainerImplC<SampleIndex>::updateUsedSortedSamples_(size_t j) const
{
    const SampleIndex* p = data(usedSortedSamples_[j]);
    const SampleIndex* p_end = data(usedSortedSamples_[j]) + size(usedSortedSamples_[j]);
    SampleIndex* q = &usedSortedSamples_[j][0];

    const SampleIndex* pSampleStatus = data(sampleStatus_);

    while (p != p_end) {
        SampleIndex i = *p;
        *q = i;
        ++p;
        q += (pSampleStatus[i] != UNUSED);
    }

    usedSortedSamples_[j].resize(q - data(usedSortedSamples_[j]));
}


template<typename SampleIndex>
void TreeTrainerImplC<SampleIndex>::updateSampleStatus_(
    const TreeNode* parentNodes, 
    const TreeNode* childNodes
) const
{
    SampleIndex* pSampleStatus = &sampleStatus_[0];

    for (size_t i = 0; i < sampleCount_; ++i) {

        size_t s = pSampleStatus[i];
        if (s == UNUSED) continue;

        const TreeNode* node = parentNodes + s;
        if (node->isLeaf) {
            pSampleStatus[i] = UNUSED;
            continue;
        }
        
        node = (inData_(i, node->j) < node->x) ? node->leftChild : node->rightChild;
        s = node - childNodes;
        pSampleStatus[i] = static_cast<SampleIndex>(s);
    }
}


template<typename SampleIndex>
void TreeTrainerImplC<SampleIndex>::initOrderedSamples_(size_t j, const vector<size_t>& sampleCountByNode) const
{
    size_t nodeCount = size(sampleCountByNode);
    orderedSamples_.resize(size(usedSortedSamples_[j]));
    vector<SampleIndex*> pSortedSamplesByNode(nodeCount);
    SampleIndex* s = data(orderedSamples_);
    for (size_t k = 0; k < nodeCount; ++k) {
        pSortedSamplesByNode[k] = s;
        s += sampleCountByNode[k];
    }

    auto pUsedSortedSamples = cbegin(usedSortedSamples_[j]);
    auto pUsedSortedSamplesEnd = cend(usedSortedSamples_[j]);
    auto l = begin(pSortedSamplesByNode);
    auto pNodeIndex = cbegin(sampleStatus_);
    while (pUsedSortedSamples != pUsedSortedSamplesEnd) {
        SampleIndex i = *(pUsedSortedSamples++);
        *(pSortedSamplesByNode[pNodeIndex[i]]++) = i;
    }
}

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImplC<uint8_t>;
template class TreeTrainerImplC<uint16_t>;
template class TreeTrainerImplC<uint32_t>;
template class TreeTrainerImplC<uint64_t>;

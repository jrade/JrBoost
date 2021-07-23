//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImplB.h"
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
TreeTrainerImplB<SampleIndex>::TreeTrainerImplB(CRefXXf inData, CRefXs strata) :
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
vector<vector<SampleIndex>> TreeTrainerImplB<SampleIndex>::createSortedSamples_() const
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
unique_ptr<BasePredictor> TreeTrainerImplB<SampleIndex>::train(
    CRefXd outData, CRefXd weights, const TreeOptions& options) const
{
    // profiling zero calibration
    PROFILE::PUSH(PROFILE::ZERO);
    size_t ITEM_COUNT = 1;

    // validate data
    //PROFILE::SWITCH(ITEM_COUNT, PROFILE::VALIDATE);
    //validateData_(outData, weights);
    //ITEM_COUNT = sampleCount_;

    // initialize used variables
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::USED_VARIABLES);
    ITEM_COUNT = initUsedVariables_(options);

    // initialize sample status
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SAMPLE_STATUS);
    ITEM_COUNT = sampleCount_;
    size_t activeSampleCount = initSampleStatus_(options, weights);
    vector<size_t> sampleCountByStatus = { sampleCount_ - activeSampleCount, activeSampleCount };

    // initialize tree
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_TREE);
    ITEM_COUNT = 1;
    vector<vector<TreeNode>> nodes(options.maxDepth() + 1);
    nodes.front().resize(1);
    vector<NodeBuilder<SampleIndex>> nodeBuilders;

    size_t parentCount = 1;

    size_t d = 0;
    while (true) {

        // find best splits

        nodeBuilders.resize(parentCount);
        for (size_t k = 0; k < parentCount; ++k)
            nodeBuilders[k].reset(inData_, outData, weights, options);

        for (size_t j : usedVariables_) {

            PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES_C);
            ITEM_COUNT = sampleCount_;
            initOrderedSamples_(j, sampleCountByStatus);

            PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_NODE_BUILDER);
            ITEM_COUNT = activeSampleCount;
            const SampleIndex* p = data(sampleBuffer_) + sampleCountByStatus[0];
            for (size_t k = 0; k < parentCount; ++k) {
                nodeBuilders[k].update(
                    j,
                    p,
                    p + sampleCountByStatus[k + 1]);
                p += sampleCountByStatus[k + 1];
            }
            ASSERT(p == data(sampleBuffer_) + size(sampleBuffer_));
        }

        // create child nodes

        size_t maxChildCount = 2 * parentCount;
        nodes[d + 1].resize(maxChildCount);
        TreeNode* pParents = data(nodes[d]);
        TreeNode* pChildren = data(nodes[d + 1]);
        sampleCountByStatus.resize(maxChildCount + 1);

        // update parent and init child nodes

        TreeNode* p = pParents;
        TreeNode* c = pChildren;
        size_t* n = data(sampleCountByStatus) + 1;
        for (size_t k = 0; k < parentCount; ++k)
            nodeBuilders[k].initNodes(&p, &c, &n);
        size_t childCount = c - pChildren;
        nodes[d + 1].resize(childCount);
        sampleCountByStatus.resize(childCount + 1);
        activeSampleCount = std::accumulate(
            cbegin(sampleCountByStatus) + 1, cend(sampleCountByStatus), static_cast<size_t>(0));
        sampleCountByStatus[0] = sampleCount_ - activeSampleCount;

        d += 1;
        if (d == options.maxDepth() || childCount == 0) break;

        // update sample status

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SAMPLE_STATUS);
        ITEM_COUNT = sampleCount_;
        updateSampleStatus_(pParents, pChildren);

        parentCount = childCount;
    }

    PROFILE::POP(ITEM_COUNT);

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
void TreeTrainerImplB<SampleIndex>::validateData_(CRefXd outData, CRefXd weights) const
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
size_t TreeTrainerImplB<SampleIndex>::initUsedVariables_(const TreeOptions& options) const
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
size_t TreeTrainerImplB<SampleIndex>::initSampleStatus_(const TreeOptions& options, CRefXd weights) const
{
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
            for (auto p = begin(sampleStatus_); p != end(sampleStatus_); ++p) {
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
        // these samples are stored in sampleBuffer_
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
                bool b = BernoulliDistribution(k, n) (rne_);
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
                bool b = BernoulliDistribution(k[stratum], n[stratum]) (rne_);
                pSampleStatus[i] = b;
                k[stratum] -= b;
                --n[stratum];
            }
        }
    }

    return usedSampleCount;
}


template<typename SampleIndex>
void TreeTrainerImplB<SampleIndex>::updateSampleStatus_(
    const TreeNode* parentNodes,
    const TreeNode* childNodes
) const
{
    SampleIndex* pSampleStatus = data(sampleStatus_);

    for (size_t i = 0; i < sampleCount_; ++i) {

        size_t s = pSampleStatus[i];
        if (s == 0) continue;

        const TreeNode* node = parentNodes + s - 1;
        if (node->isLeaf) {
            pSampleStatus[i] = 0;
            continue;
        }

        node = (inData_(i, node->j) < node->x) ? node->leftChild : node->rightChild;
        s = node - childNodes + 1;
        pSampleStatus[i] = static_cast<SampleIndex>(s);
    }
}


template<typename SampleIndex>
void TreeTrainerImplB<SampleIndex>::initOrderedSamples_(size_t j, const vector<size_t>& sampleCountByStatus) const
{
    size_t statusCount = size(sampleCountByStatus);
    ASSERT(statusCount >= 2);
    if (statusCount == 2)
        return initOrderedSamplesFast_(j, sampleCountByStatus);

    vector<SampleIndex*> pSortedSamplesByStatus(statusCount);
    sampleBuffer_.resize(sampleCount_);
    SampleIndex* p = data(sampleBuffer_);
    for (size_t k = 0; k < statusCount; ++k) {
        pSortedSamplesByStatus[k] = p;
        p += sampleCountByStatus[k];
    }

    auto pSortedSamples = cbegin(sortedSamples_[j]);
    auto pSortedSamplesEnd = cend(sortedSamples_[j]);
    auto l = begin(pSortedSamplesByStatus);
    auto pStatus = cbegin(sampleStatus_);
    while (pSortedSamples != pSortedSamplesEnd) {
        SampleIndex i = *(pSortedSamples++);
        *(pSortedSamplesByStatus[pStatus[i]]++) = i;
    }
}


template<typename SampleIndex>
void TreeTrainerImplB<SampleIndex>::initOrderedSamplesFast_(size_t j, const vector<size_t>& sampleCountByStatus) const
{
    sampleBuffer_.resize(sampleCount_);

    const SampleIndex* p = data(sortedSamples_[j]);

    SampleIndex* q = data(sampleBuffer_) + sampleCountByStatus[0];
    SampleIndex* q_end = q + sampleCountByStatus[1];
    ASSERT(q_end == data(sampleBuffer_) + sampleCount_);

    const SampleIndex* pSampleStatus = data(sampleStatus_);

    while (q != q_end) {
        SampleIndex i = *p;
        *q = i;
        ++p;
        q += pSampleStatus[i];
    }
}

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImplB<uint8_t>;
template class TreeTrainerImplB<uint16_t>;
template class TreeTrainerImplB<uint32_t>;
template class TreeTrainerImplB<uint64_t>;

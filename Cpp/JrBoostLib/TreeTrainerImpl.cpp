//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImpl.h"
#include "StumpOptions.h"
#include "pdqsort.h"


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
    CRefXd outData, CRefXd weights, const StumpOptions& options) const
{
    PROFILE::PUSH(PROFILE::ZERO);
    size_t ITEM_COUNT = 1;

    //PROFILE::SWITCH(ITEM_COUNT, PROFILE::VALIDATE);
    //validateData_(outData, weights);
    //ITEM_COUNT = sampleCount_;

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::USED_SAMPLES);
    ITEM_COUNT = sampleCount_;

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::USED_VARIABLES);
    const size_t candidateVariableCount = initUsedVariables_(options);
    ITEM_COUNT = candidateVariableCount;

    vector<vector<TreePredictor::Node>> nodes(options.maxDepth() + 1);
    nodes[0].resize(1);

    vector<size_t> sampleCountByStatus = initSampleStatus_(options, weights);
    size_t d = 0;

    while(true) {

        size_t parentCount = nodes[d].size();
       
        nodeBuilders_.resize(std::max(parentCount + 1, nodeBuilders_.size()));
        for (size_t k = 1; k < parentCount + 1; ++k)
            nodeBuilders_[k].reset();

        for (size_t j : usedVariables_) {

            PROFILE::SWITCH(ITEM_COUNT, PROFILE::SORTED_USED_SAMPLES);
            initSortedSamplesByStatus_(sampleCountByStatus, j);
            ITEM_COUNT = sampleCount_;

            PROFILE::SWITCH(ITEM_COUNT, PROFILE::BEST_SPLIT);
            for (size_t k = 1; k < parentCount + 1; ++k)
                nodeBuilders_[k].update(j, inData_, outData, weights, options);
            ITEM_COUNT = sampleCountByStatus[1];
        }

        size_t maxChildCount = 2 * parentCount;
        nodes[d + 1].resize(maxChildCount);
        TreePredictor::Node* pParents = &nodes[d][0];
        TreePredictor::Node* pChildren = &nodes[d + 1][0];

        TreePredictor::Node* p = pParents;
        TreePredictor::Node* c = pChildren;
        for (size_t k = 1; k < parentCount + 1; ++k)
            nodeBuilders_[k].initNodes(&p, &c);
        size_t childCount = c - pChildren;
        nodes[d + 1].resize(childCount);

        d += 1;
        if (d == options.maxDepth() || childCount == 0) break;

        sampleCountByStatus = updateSampleStatus_(pParents, pChildren);

        //if (omp_get_thread_num() == 0) {
        //    for (size_t n : sampleCountByStatus)
        //        std::cout << n << ' ';
        //    std::cout << std::endl;
        //}
    }

    nodes.resize(d + 1);

    //if (omp_get_thread_num() == 0)
    //    std::cout << std::endl;


    if (omp_get_thread_num() == 0) {
        PROFILE::SPLIT_ITERATION_COUNT += nodeBuilders_[1].iterationCount();
        PROFILE::SLOW_BRANCH_COUNT += nodeBuilders_[1].slowBranchCount();
    }

    PROFILE::POP(ITEM_COUNT);

    return std::make_unique<TreePredictor>(&nodes[0][0], std::move(nodes));
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
    if ((weights < 0.0).any())
        throw std::invalid_argument("Train weights have negative values.");
}


template<typename SampleIndex>
vector<size_t> TreeTrainerImpl<SampleIndex>::initSampleStatus_(const StumpOptions& options, CRefXd weights) const
{
    size_t usedSampleCount;
    sampleStatus_.resize(sampleCount_);

    double minSampleWeight = options.minAbsSampleWeight();
    double minRelSampleWeight = options.minRelSampleWeight();
    if (minRelSampleWeight > 0.0)
        minSampleWeight = std::max(minSampleWeight, weights.maxCoeff() * minRelSampleWeight);

    bool isStratified = options.isStratified();

    if (minSampleWeight == 0.0) {

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
        // these samples are stored in tmpSamples
        // then we select among those samples

        if (!isStratified) {
            tmpSamples_.resize(sampleCount_);
            auto p = begin(tmpSamples_);
            for (size_t i = 0; i < sampleCount_; ++i) {
                sampleStatus_[i] = 0;
                bool b = weights(i) >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
            }
            tmpSamples_.resize(p - begin(tmpSamples_));

            size_t n = tmpSamples_.size();
            size_t k = static_cast<size_t>(options.usedSampleRatio() * n + 0.5);
            if (k == 0 && n > 0) k = 1;
            usedSampleCount = k;

            for (SampleIndex i : tmpSamples_) {
                bool b = BernoulliDistribution(k, n) (rne_);
                sampleStatus_[i] = b;
                k -= b;
                --n;
            }
        }

        else {
            tmpSamples_.resize(sampleCount_);
            array<size_t, 2> n = { 0, 0 };
            auto p = begin(tmpSamples_);
            for (size_t i = 0; i < sampleCount_; ++i) {
                size_t stratum = strata_[i];
                bool b = weights(i) >= minSampleWeight;
                *p = static_cast<SampleIndex>(i);
                p += b;
                n[stratum] += b;
                sampleStatus_[i] = 0;
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
                sampleStatus_[i] = b;
                k[stratum] -= b;
                --n[stratum];
            }
        }
    }

    return { sampleCount_ - usedSampleCount, usedSampleCount };
}


template<typename SampleIndex>
size_t TreeTrainerImpl<SampleIndex>::initUsedVariables_(const StumpOptions& options) const
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
void TreeTrainerImpl<SampleIndex>::initSortedSamplesByStatus_(const vector<size_t>& sampleCountByStatus, size_t j) const
{
    size_t maxStatus = sampleCountByStatus.size();

    vector<vector<SampleIndex>::iterator> sortedSamplesByStatus(maxStatus);
    for (size_t s = 0; s < maxStatus; ++s) {
        vector<SampleIndex>& sortedSamples = nodeBuilders_[s].sortedSamples();
        sortedSamples.resize(sampleCountByStatus[s]);
        sortedSamplesByStatus[s] = begin(sortedSamples);
    }

    auto p = begin(sortedSamplesByStatus);
    auto r = cbegin(sampleStatus_);
    auto q0 = cbegin(sortedSamples_[j]);
    auto q1 = cend(sortedSamples_[j]);
    while (q0 != q1) {
        SampleIndex i = *(q0++);
        *(p[r[i]]++) = i;
    }
}


template<typename SampleIndex>
vector<size_t> TreeTrainerImpl<SampleIndex>::updateSampleStatus_(
    const TreePredictor::Node* parentNodes, const TreePredictor::Node* childNodes
) const
{
    uint8_t* pSampleStatus = &sampleStatus_[0];

    for (size_t i = 0; i < sampleCount_; ++i) {
        uint8_t s = pSampleStatus[i];
        if (s == 0) continue;
        const TreePredictor::Node* node = parentNodes + s - 1;
        if (node->isLeaf)
            pSampleStatus[i] = 0;
        else {
            node = (inData_(i, node->j) < node->x) ? node->leftChild : node->rightChild;
            pSampleStatus[i] = static_cast<uint8_t>(node - childNodes + 1);
        }
    }

    size_t maxStatus = *std::max_element(cbegin(sampleStatus_), cend(sampleStatus_));
    vector<size_t> sampleCountByStatus(maxStatus + 1, 0);
    for (uint8_t s : sampleStatus_)
        ++sampleCountByStatus[s];
    return sampleCountByStatus;
}

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImpl<uint8_t>;
template class TreeTrainerImpl<uint16_t>;
template class TreeTrainerImpl<uint32_t>;
template class TreeTrainerImpl<uint64_t>;

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeTrainerImplA.h"
#include "TreeOptions.h"
#include "StumpPredictor.h"
#include "TreePredictor.h"
#include "TrivialPredictor.h"
#include "pdqsort.h"

#ifdef _MSC_VER
    #pragma warning(disable: 4701)
#endif

template<typename SampleIndex>
TreeTrainerImplA<SampleIndex>::TreeTrainerImplA(CRefXXf inData, CRefXs strata) :
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
vector<vector<SampleIndex>> TreeTrainerImplA<SampleIndex>::createSortedSamples_() const
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
unique_ptr<BasePredictor> TreeTrainerImplA<SampleIndex>::train(
    CRefXd outData, CRefXd weights, const TreeOptions& options) const
{
    ASSERT(options.maxDepth() == 1);

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

    // initialize sample mask
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SAMPLE_MASK);
    ITEM_COUNT = sampleCount_;
    const size_t usedSampleCount = initUsedSampleMask_(options, weights);

    SplitData splitData;
    splitData.isInit = false;
    for (size_t j : usedVariables_) {

        // initialize sorted used samples
        PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_USED_SORTED_SAMPLES);
        ITEM_COUNT = sampleCount_;
        initSortedUsedSamples_(usedSampleCount, j);

        // find best split
        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_NODE_BUILDER);
        const SampleIndex* usedSamples = data(sampleBuffer_);
        ITEM_COUNT = usedSampleCount;
        updateSplit_(&splitData, outData, weights, options, usedSamples, usedSampleCount, j);
    }

    PROFILE::POP(ITEM_COUNT);

    if (omp_get_thread_num() == 0) {
        PROFILE::SPLIT_ITERATION_COUNT += splitData.iterationCount;
        PROFILE::SLOW_BRANCH_COUNT += splitData.slowBranchCount;
    }

#if 1

    if (!splitData.splitFound) {
        double y = (splitData.sumW == 0.0) ? 0.0 : splitData.sumWY / splitData.sumW;
        return std::make_unique<TrivialPredictor>(y);
    }

    return std::make_unique<StumpPredictor>(
        static_cast<uint32_t>(splitData.j),
        splitData.x,
        static_cast<float>(splitData.leftY),
        static_cast<float>(splitData.rightY));

#else
    vector<vector<TreeNode>> nodes(options.maxDepth() + 1);
    nodes[0].resize(1);
    TreeNode* root = data(nodes[0]);
    double y = (splitData.sumW == 0.0) ? 0.0 : splitData.sumWY / splitData.sumW;
    root->y = static_cast<float>(y);
    if (!splitData.splitFound)
        root->isLeaf = true;
    else {
        nodes[1].resize(2);
        root->isLeaf = false;
        root->j = static_cast<uint32_t>(splitData.j);
        root->x = splitData.x;
        root->leftChild = data(nodes[1]);
        root->leftChild->isLeaf = true;
        root->leftChild->y = static_cast<float>(splitData.leftY);
        root->rightChild = root->leftChild + 1;
        root->rightChild->isLeaf = true;
        root->rightChild->y = static_cast<float>(splitData.rightY);
    }
    return std::make_unique<TreePredictor>(root, move(nodes));
#endif
};


template<typename SampleIndex>
void TreeTrainerImplA<SampleIndex>::validateData_(CRefXd outData, CRefXd weights) const
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
size_t TreeTrainerImplA<SampleIndex>::initUsedVariables_(const TreeOptions& options) const
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
size_t TreeTrainerImplA<SampleIndex>::initUsedSampleMask_(const TreeOptions& options, CRefXd weights) const
{
    const size_t minNodeSize = options.minNodeSize();

    size_t usedSampleCount;
    usedSampleMask_.resize(sampleCount_);

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

            for (auto p = begin(usedSampleMask_); p != end(usedSampleMask_); ++p) {
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
            for (auto p = begin(usedSampleMask_); p != end(usedSampleMask_); ++p) {
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

        sampleBuffer_.resize(sampleCount_);
        usedSampleMask_.assign(sampleCount_, 0);
        uint8_t* pSampleMask = data(usedSampleMask_);

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
                pSampleMask[i] = b;
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
                pSampleMask[i] = b;
                k[stratum] -= b;
                --n[stratum];
            }
        }
    }

    return usedSampleCount;
}


template<typename SampleIndex>
void TreeTrainerImplA<SampleIndex>::initSortedUsedSamples_(size_t usedSampleCount, size_t j) const
{
    sampleBuffer_.resize(usedSampleCount);

    auto p0 = cbegin(sortedSamples_[j]);
    auto q0 = begin(sampleBuffer_);
    auto q1 = end(sampleBuffer_);
    auto m = cbegin(usedSampleMask_);

    while (q0 != q1) {
        // branch-free implementation of copy_if
        SampleIndex i = *p0;
        *q0 = i;
        ++p0;
        q0 += m[i];
    }
}


template<typename SampleIndex>
void TreeTrainerImplA<SampleIndex>::updateSplit_(
    typename TreeTrainerImplA<SampleIndex>::SplitData* splitData,
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
        splitData->score = sumWY * sumWY / sumW;
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

//----------------------------------------------------------------------------------------------------------------------

template class TreeTrainerImplA<uint8_t>;
template class TreeTrainerImplA<uint16_t>;
template class TreeTrainerImplA<uint32_t>;
template class TreeTrainerImplA<uint64_t>;

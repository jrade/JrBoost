//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "StumpTrainerImpl.h"

#include "pdqsort.h"
#include "StumpPredictor.h"
#include "TreeOptions.h"
#include "TrivialPredictor.h"


template<typename SampleIndex>
StumpTrainerImpl<SampleIndex>::StumpTrainerImpl(CRefXXfc inData, CRefXs strata) :
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
vector<vector<SampleIndex>> StumpTrainerImpl<SampleIndex>::createSortedSamples_() const
{
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
    return sortedSamples;
}

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
unique_ptr<BasePredictor> StumpTrainerImpl<SampleIndex>::train(
    CRefXd outData, CRefXd weights, const TreeOptions& options, size_t threadCount) const
{
    (void)threadCount;

    ASSERT(options.maxDepth() == 1);

    // profiling zero calibration
    PROFILE::PUSH(PROFILE::ZERO);
    size_t ITEM_COUNT = 1;

    // validate data
    //PROFILE::SWITCH(ITEM_COUNT, PROFILE::VALIDATE);
    //validateData_(outData, weights);
    //ITEM_COUNT = sampleCount_;

    // initialize used variables
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_USED_VARIABLES);
    ITEM_COUNT = initUsedVariables_(options);

    // initialize sample mask
    PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_SAMPLE_STATUS);
    ITEM_COUNT = sampleCount_;
    const size_t usedSampleCount = initUsedSampleMask_(options, weights);


    // prepare for finding best split ..........................................

    const size_t minNodeSize = options.minNodeSize();
    size_t slowBranchCount = 0;

    size_t bestJ = 0;
    float bestX = 0.0f;
    double bestLeftY = 0.0;
    double bestRightY = 0.0;
    bool splitFound = false;

    // the following variables are initialized during the first iteration below
    bool sumsInit = false;
    double sumW = 0.0;
    double sumWY = 0.0;
    double bestScore = 0.0;
    double tol = 0.0;   // estimate of the rounding off error we can expect in rightSumW towards the end of the loop
    double minNodeWeight = 0.0;

    for (size_t j : usedVariables_) {

        // initialize sorted used samples ......................................

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::INIT_ORDERED_SAMPLES);
        ITEM_COUNT = sampleCount_;

        initSortedUsedSamples_(usedSampleCount, j);


        // initialize sums .........................................................

        if (!sumsInit) {
            PROFILE::SWITCH(ITEM_COUNT, PROFILE::SUMS);
            ITEM_COUNT = size(sampleBuffer_);

            std::tie(sumW, sumWY) = initSums_(outData, weights);
            if (sumW == 0) {
                PROFILE::POP(ITEM_COUNT);
                return std::make_unique<TrivialPredictor>(0.0);
            }

            bestScore = sumWY * sumWY / sumW + options.minGain();
            tol = sumW * std::sqrt(static_cast<double>(usedSampleCount)) * numeric_limits<double>::epsilon() / 2;
            minNodeWeight = std::max<double>(options.minNodeWeight(), tol);

            sumsInit = true;
        }


        // find best split .....................................................

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::UPDATE_SPLITS);
        ITEM_COUNT = usedSampleCount;

        double leftSumW = 0.0;
        double leftSumWY = 0.0;
        double rightSumW = sumW;
        double rightSumWY = sumWY;

        const auto pBegin = cbegin(sampleBuffer_);
        const auto pEnd = cend(sampleBuffer_);
        auto p = pBegin;
        size_t nextI = *p;

        // this is where most execution time is spent ......
            
        while (p != pEnd - 1) {

            const size_t i = nextI;
            nextI = *++p;
            const double w = weights(i);
            const double y = outData(i);
            leftSumW += w;
            rightSumW -= w;
            leftSumWY += w * y;
            rightSumWY -= w * y;
            const double score = leftSumWY * leftSumWY / leftSumW + rightSumWY * rightSumWY / rightSumW;

            if (score <= bestScore) continue;  // usually true

        //..................................................

            ++slowBranchCount;

            if (p < pBegin + minNodeSize
                || p > pEnd - minNodeSize
                || leftSumW < minNodeWeight
                || rightSumW < minNodeWeight
            ) continue;

            const float leftX = inData_(i, j);
            const float rightX = inData_(nextI, j);
            const float midX = (leftX + rightX) / 2;
    
            if (leftX == midX) continue;

            bestScore = score;
            bestJ = j;
            bestX = midX;
            bestLeftY = leftSumWY / leftSumW;
            bestRightY = rightSumWY / rightSumW;
            splitFound = true;
        }
    }

    PROFILE::POP(ITEM_COUNT);

    const size_t usedVariableCount = size(usedVariables_);
    const size_t iterationCount = usedVariableCount * usedSampleCount;
    PROFILE::UPDATE_BRANCH_STATISTICS(iterationCount, slowBranchCount);

    if (!splitFound)
        return std::make_unique<TrivialPredictor>(sumWY / sumW);
    
    float gain = static_cast<float>(bestScore - sumWY * sumWY / sumW);
    return std::make_unique<StumpPredictor>(
        bestJ, bestX, static_cast<float>(bestLeftY), static_cast<float>(bestRightY),
        static_cast<float>(gain));
};


template<typename SampleIndex>
void StumpTrainerImpl<SampleIndex>::validateData_(CRefXd outData, CRefXd weights) const
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
size_t StumpTrainerImpl<SampleIndex>::initUsedVariables_(const TreeOptions& options) const
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
size_t StumpTrainerImpl<SampleIndex>::initUsedSampleMask_(const TreeOptions& options, CRefXd weights) const
{
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

            const size_t* s = std::data(strata_);
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
void StumpTrainerImpl<SampleIndex>::initSortedUsedSamples_(size_t usedSampleCount, size_t j) const
{
    sampleBuffer_.resize(usedSampleCount);

    auto p0 = cbegin(sortedSamples_[j]);
    auto q0 = begin(sampleBuffer_);
    auto q1 = end(sampleBuffer_);
    auto m = cbegin(usedSampleMask_);

    while (q0 != q1) {
        SampleIndex i = *p0;
        *q0 = i;
        ++p0;
        q0 += m[i];
    }
}


template<typename SampleIndex>
pair<double, double> StumpTrainerImpl<SampleIndex>::initSums_(CRefXd outData, CRefXd weights) const
{
    double sumW = 0.0;
    double sumWY = 0.0;
    for (size_t i : sampleBuffer_) {
        double w = weights(i);
        double y = outData(i);
        sumW += w;
        sumWY += w * y;
    }
    return std::make_pair(sumW, sumWY);
}

//----------------------------------------------------------------------------------------------------------------------

template class StumpTrainerImpl<uint8_t>;
template class StumpTrainerImpl<uint16_t>;
template class StumpTrainerImpl<uint32_t>;
template class StumpTrainerImpl<uint64_t>;

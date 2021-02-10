#include "pch.h"
#include "StumpTrainerImpl.h"
#include "StumpOptions.h"
#include "StumpPredictor.h"
#include "TrivialPredictor.h"


template<typename SampleIndex>
StumpTrainerImpl<SampleIndex>::StumpTrainerImpl(CRefXXf inData, RefXs strata) :
    inData_{ inData },
    sampleCount_{ static_cast<size_t>(inData.rows()) },
    variableCount_{ static_cast<size_t>(inData.cols()) },
    sortedSamples_{ createSortedSamples_() },
    strata_{ strata },
    stratum0Count_{ (strata == 0).cast<size_t>().sum() },
    stratum1Count_{ (strata == 1).cast<size_t>().sum() }
{
    ASSERT(inData.rows() != 0);
    ASSERT(inData.cols() != 0);
    ASSERT(static_cast<size_t>(strata.rows()) == sampleCount_);

    ASSERT((inData > -numeric_limits<float>::infinity()).all());
    ASSERT((inData < numeric_limits<float>::infinity()).all());
    ASSERT((strata < 2).all());
}


template<typename SampleIndex>
vector<vector<SampleIndex>> StumpTrainerImpl<SampleIndex>::createSortedSamples_() const
{
    vector<vector<SampleIndex>> sortedSamples(variableCount_);
    vector<pair<float, SampleIndex>> tmp{ sampleCount_ };
    for (size_t j = 0; j < variableCount_; ++j) {
        for (size_t i = 0; i < sampleCount_; ++i)
            tmp[i] = { inData_(i,j), static_cast<SampleIndex>(i) };
        std::sort(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return x.first < y.first; });
        sortedSamples[j].resize(sampleCount_);
        for (size_t i = 0; i < sampleCount_; ++i)
            sortedSamples[j][i] = tmp[i].second;
    }

    return sortedSamples;
}

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
unique_ptr<AbstractPredictor> StumpTrainerImpl<SampleIndex>::train(
    CRefXd outData, CRefXd weights, const StumpOptions& options) const
{
    // validate data ...........................................................

    PROFILE::PUSH(PROFILE::VALIDATE);
    size_t ITEM_COUNT = sampleCount_;

    ASSERT(static_cast<size_t>(outData.rows()) == sampleCount_);
    ASSERT((outData > -numeric_limits<double>::infinity()).all());
    ASSERT((outData < numeric_limits<double>::infinity()).all());

    ASSERT(static_cast<size_t>(weights.rows()) == sampleCount_);
    ASSERT((weights >= 0.0).all());
    ASSERT((weights < numeric_limits<double>::infinity()).all());

 
    // initialize used sample mask .............................................

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::USED_SAMPLES);
    ITEM_COUNT = sampleCount_;

    const size_t usedSampleCount = initUsedSampleMask_(options);


    // initialize used variables ...............................................

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::USED_VARIABLES);

    const size_t candidateVariableCount = initUsedVariables_(options);
    const size_t usedVariableCount = usedVariables_.size();

    ITEM_COUNT = candidateVariableCount;


    // initialize sums .........................................................

    PROFILE::SWITCH(ITEM_COUNT, PROFILE::SUMS);
    ITEM_COUNT = sampleCount_;

    initSums_(outData, weights);

    if (sumW_ == 0) {
        PROFILE::POP(ITEM_COUNT);
        cout << "Warning: sumW = 0" << endl;
        return std::make_unique<TrivialPredictor>(variableCount_, 0.0);
    }


    // prepare for finding best split ..........................................

    double bestScore = sumWY_ * sumWY_ / sumW_;
    size_t bestJ = 0;
    float bestX = 0.0f;
    double bestLeftY = 0.0;
    double bestRightY = 0.0;
    bool splitFound = false;

    const size_t minNodeSize = options.minNodeSize();
    const double tol = sumW_ * sqrt(static_cast<double>(usedSampleCount)) * numeric_limits<double>::epsilon() / 2;
    // tol = estimate of the rounding off error we can expect in rightSumW towards the end of the loop
    const double minNodeWeight = std::max<double>(options.minNodeWeight(), tol);

    size_t slowBranchCount = 0;
        
    for (size_t j : usedVariables_) {

        // initialize sorted used samples ......................................

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::SORTED_USED_SAMPLES);
        ITEM_COUNT = sampleCount_;

        initSortedUsedSamples_(usedSampleCount, j);


        // find best split .....................................................

        PROFILE::SWITCH(ITEM_COUNT, PROFILE::BEST_SPLIT);
        ITEM_COUNT = usedSampleCount;

        double leftSumW = 0.0;
        double leftSumWY = 0.0;
        double rightSumW = sumW_;
        double rightSumWY = sumWY_;

        const auto pBegin = cbegin(sortedUsedSamples_);
        const auto pEnd = cend(sortedUsedSamples_);
        auto p = pBegin;
        size_t nextI = *p;

        // this is where most execution time is spent ......
            
        while (p != pEnd - 1) {

            const size_t i = nextI;
            nextI = *++p;
            const double w = weights[i];
            const double y = outData[i];
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
#pragma omp master
    {
        PROFILE::SPLIT_ITERATION_COUNT += usedVariableCount * usedSampleCount;
        PROFILE::SLOW_BRANCH_COUNT += slowBranchCount;
    }

    if (!splitFound)
        return std::make_unique<TrivialPredictor>(variableCount_, sumWY_ / sumW_);
    
    return unique_ptr<StumpPredictor>(new StumpPredictor(variableCount_, bestJ, bestX, bestLeftY, bestRightY));
};


template<typename SampleIndex>
size_t StumpTrainerImpl<SampleIndex>::initUsedSampleMask_(const StumpOptions& options) const
{
    size_t usedSampleCount;
    usedSampleMask_.resize(sampleCount_);

    if (!options.isStratified()) {
        usedSampleCount = static_cast<size_t>(options.usedSampleRatio() * sampleCount_ + 0.5);
        if (usedSampleCount == 0) usedSampleCount = 1;

        size_t n = sampleCount_;
        size_t k = usedSampleCount;
        // create a random mask of length n with k ones and n - k zeros
        for (auto p = begin(usedSampleMask_); p != end(usedSampleMask_); ++p) {
            bool b = BernoulliDistribution(k, n) (rne_);
            *p = b;
            k -= b;
            --n;
        }
    }

    else {
        // same as above, but done per stratum

        array<size_t, 2> n{ stratum0Count_, stratum1Count_ };

        size_t usedStratum0Count = static_cast<size_t>(options.usedSampleRatio() * stratum0Count_ + 0.5);
        if (usedStratum0Count == 0 && stratum0Count_ > 0) usedStratum0Count = 1;
        size_t usedStratum1Count = static_cast<size_t>(options.usedSampleRatio() * stratum1Count_ + 0.5);
        if (usedStratum1Count == 0 && stratum1Count_ > 0) usedStratum1Count = 1;
        array<size_t, 2> k{ usedStratum0Count, usedStratum1Count };

        const size_t* s = &strata_(0);
        for (auto p = begin(usedSampleMask_); p != end(usedSampleMask_); ++p) {
            size_t stratum = *s;
            ++s;
            bool b = BernoulliDistribution(k[stratum], n[stratum]) (rne_);
            *p = b;
            k[stratum] -= b;
            --n[stratum];
        }

        usedSampleCount = usedStratum0Count + usedStratum1Count;
    }

    return usedSampleCount;
}


template<typename SampleIndex>
size_t StumpTrainerImpl<SampleIndex>::initUsedVariables_(const StumpOptions& options) const
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
void StumpTrainerImpl<SampleIndex>::initSums_(const CRefXd& outData, const CRefXd& weights) const
{
    sumW_ = 0.0;
    sumWY_ = 0.0;
    for (size_t i = 0; i < sampleCount_; ++i) {
        double m = usedSampleMask_[i];
        double w = weights[i];
        double y = outData[i];
        sumW_ += m * w;
        sumWY_ += m * w * y;
    }
}


template<typename SampleIndex>
void StumpTrainerImpl<SampleIndex>::initSortedUsedSamples_(size_t usedSampleCount, size_t j) const
{
    sortedUsedSamples_.resize(usedSampleCount);

    auto p0 = cbegin(sortedSamples_[j]);
    auto q0 = begin(sortedUsedSamples_);
    auto q1 = end(sortedUsedSamples_);

    while (q0 != q1) {
        SampleIndex i = *p0;
        *q0 = i;
        ++p0;
        q0 += usedSampleMask_[i];
    }
}

//----------------------------------------------------------------------------------------------------------------------

template class StumpTrainerImpl<uint8_t>;
template class StumpTrainerImpl<uint16_t>;
template class StumpTrainerImpl<uint32_t>;
template class StumpTrainerImpl<uint64_t>;

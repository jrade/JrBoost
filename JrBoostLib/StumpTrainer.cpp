#include "pch.h"
#include "StumpTrainer.h"
#include "StumpOptions.h"
#include "TrivialPredictor.h"

StumpTrainer::StumpTrainer() :
    options_{ std::make_unique<StumpOptions>() }
{
}

void StumpTrainer::setInData(CRefXXf inData)
{ 
    ASSERT(inData.isFinite().all());
    assign(inData_, inData);
    sampleCount_ = inData.rows();
    variableCount_ = inData.cols();

    sortedSamples_.resize(variableCount_);
    vector<std::pair<float, size_t>> tmp{ sampleCount_ };
    for (size_t j = 0; j < variableCount_; ++j) {
        for (size_t i = 0; i < sampleCount_; ++i)
            tmp[i] = { inData(i,j), i };
        std::sort(begin(tmp), end(tmp));
        sortedSamples_[j].resize(sampleCount_);
        for (size_t i = 0; i < sampleCount_; ++i)
            sortedSamples_[j][i] = tmp[i].second;
    }
}

void StumpTrainer::setOutData(const ArrayXf& outData)
{
    ASSERT(outData.isFinite().all());
    outData_ = outData;
}

void StumpTrainer::setWeights(const ArrayXf& weights)
{
    ASSERT((weights < numeric_limits<float>::infinity()).all());
    ASSERT((weights >= 0).all());
    weights_ = weights;
}

void StumpTrainer::setOptions(const AbstractOptions& opt)
{
    const StumpOptions& opt1 = dynamic_cast<const StumpOptions&>(opt);
    options_.reset(opt1.clone());
}

AbstractPredictor* StumpTrainer::train() const
{
    ASSERT(sampleCount_ > 0);
    ASSERT(variableCount_ > 0);
    ASSERT(static_cast<size_t>(outData_.size()) == sampleCount_);
    ASSERT(static_cast<size_t>(weights_.size()) == sampleCount_);

    if (options_->highPrecision())
        return trainImpl_<double>();
    else
        return trainImpl_<float>();
}

template<typename F>
AbstractPredictor* StumpTrainer::trainImpl_() const
{
    size_t n = 0;

    size_t t0 = 0;
    size_t t1 = 0;
    size_t t2 = 0;

    START_TIMER(t0);

    size_t usedSampleCount = std::max(
        static_cast<size_t>(1),
        static_cast<size_t>(static_cast<double>(options_->usedSampleRatio()) * sampleCount_ + 0.5)
    );

    size_t usedVariableCount = std::max(
        static_cast<size_t>(1), 
        static_cast<size_t>(static_cast<double>(options_->usedVariableRatio()) * variableCount_ + 0.5)
    );

    // used samples mask

    usedSampleMask_.resize(sampleCount_);
    fastRandomMask(
        begin(usedSampleMask_), 
        end(usedSampleMask_), 
        usedSampleCount, 
        theRNE
    );

    // used variables

    usedVariables_.resize(variableCount_);
    std::iota(begin(usedVariables_), end(usedVariables_), 0);
    fastOrderedRandomSubset(
        begin(usedVariables_), 
        end(usedVariables_), 
        begin(usedVariables_), 
        begin(usedVariables_) + usedVariableCount, 
        theRNE
    );
 
    usedVariables_.resize(usedVariableCount);

    // sums

    F sumW = F{ 0 };
    F sumWY = F{ 0 };
    for (size_t i = 0; i < sampleCount_; ++i) {
        F m = usedSampleMask_[i];
        F w = weights_[i];
        F y = outData_[i];
        sumW += m * w;
        sumWY += m * w * y;
    }

    ASSERT(sumW != 0);

    // find best split

    F bestScore = sumWY * sumWY / sumW;
    size_t bestJ = static_cast<size_t>(-1);
    float bestX = numeric_limits<float>::quiet_NaN();
    F bestLeftY = numeric_limits<F>::quiet_NaN();
    F bestRightY = numeric_limits<F>::quiet_NaN();

    sortedUsedSamples_.resize(usedSampleCount);

    size_t minNodeSize = options_->minNodeSize();

    const F tol = sumW * sqrt(static_cast<F>(usedSampleCount)) * numeric_limits<F>::epsilon() / 2;
    // tol = estimate of the rounding off error we can expect in rightSumW towards the end of the loop
    const F minNodeWeight = std::max<F>(options_->minNodeWeight(), tol);
        
    for (size_t j : usedVariables_) {

        // prepare list of samples
        SWITCH_TIMER(t0, t1);

        fastCopyIf(
            begin(sortedSamples_[j]),
            begin(sortedUsedSamples_),
            end(sortedUsedSamples_),
            [&](size_t i) { return usedSampleMask_[i]; }
        );

        // find best split
        SWITCH_TIMER(t1, t2);

        F leftSumW = 0;
        F leftSumWY = 0;
        F rightSumW = sumW;
        F rightSumWY = sumWY;

        const auto pBegin = begin(sortedUsedSamples_);
        const auto pEnd = end(sortedUsedSamples_);
        auto p = pBegin;
        size_t nextI = *p;

        // this is where most execution time is spent ......

        while (p != pEnd - 1) {
            const size_t i = nextI;
            nextI = *++p;
            const F w = weights_[i];
            const F y = outData_[i];
            leftSumW += w;
            rightSumW -= w;
            leftSumWY += w * y;
            rightSumWY -= w * y;
            const F score = leftSumWY * leftSumWY / leftSumW + rightSumWY * rightSumWY / rightSumW;
            if (score <= bestScore) continue;  // usually true

        //..................................................

            ++n;

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
        }

        //{
        //    const F w = weights_[nextI];
        //    const F y = outData_[nextI];
        //    leftSumW += w;
        //    rightSumW -= w;
        //    leftSumWY += w * y;
        //    rightSumWY -= w * y;
        //}

        SWITCH_TIMER(t2, t0);
    }

    STOP_TIMER(t0);

    if (options_->profile()) {
        cout << t0 << endl;
        cout << t1 << " (" << static_cast<float>(t1) / (sampleCount_ * usedVariableCount) << ")" << endl;
        cout << t2 << " (" << static_cast<float>(t2) / (usedSampleCount * usedVariableCount) << ")" << endl;
        cout << 100.0f * n / ((usedSampleCount - 1) * usedVariableCount) << "%" << endl;
        cout << endl;
    }

    if (bestJ == static_cast<size_t>(-1))
        return new TrivialPredictor{ static_cast<float>(sumWY / sumW), variableCount_ };
    else
        return new StumpPredictor{ 
            variableCount_, bestJ, bestX, static_cast<float>(bestLeftY), static_cast<float>(bestRightY) };
};

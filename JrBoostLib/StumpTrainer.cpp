#include "pch.h"
#include "StumpTrainer.h"
#include "StumpOptions.h"

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
    vector<pair<float, size_t>> tmp{ sampleCount_ };
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
    ASSERT((weights < std::numeric_limits<float>::infinity()).all());
    ASSERT((weights >= 0).all());
    weights_ = weights;
}

void StumpTrainer::setOptions(const AbstractOptions& opt)
{
    const StumpOptions& opt1 = dynamic_cast<const StumpOptions&>(opt);
    options_.reset(opt1.clone());
}

StumpPredictor* StumpTrainer::train() const
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
StumpPredictor* StumpTrainer::trainImpl_() const
{
    size_t n = 0;
    size_t t0 = 0;
    size_t t1 = 0;
    size_t t2 = 0;

    size_t t = clockCycleCount();
    t0 -= t;

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

    // find best split

    F bestScore = sumWY * sumWY / sumW;
    size_t bestJ = 0;
    float bestX = -std::numeric_limits<float>::infinity();
    F bestLeftY = std::numeric_limits<F>::quiet_NaN();
    F bestRightY = sumWY / sumW;

    sortedUsedSamples_.resize(usedSampleCount);
    float minNodeWeight = std::max(options_->minNodeWeight(), std::numeric_limits<float>::min());

    for (size_t j : usedVariables_) {

        // prepare list of samples
        t = clockCycleCount();
        t0 += t;
        t1 -= t;

        fastCopyIf(
            begin(sortedSamples_[j]),
            begin(sortedUsedSamples_),
            end(sortedUsedSamples_),
            [&](size_t i) { return usedSampleMask_[i]; }
        );

        // find best split
        t = clockCycleCount();
        t1 += t;
        t2 -= t;
        
        F leftSumW = F{ 0 };
        F leftSumWY = F{ 0 };
        F rightSumW = sumW;
        F rightSumWY = sumWY;

        auto pBegin = begin(sortedUsedSamples_);
        auto pEnd = end(sortedUsedSamples_);
        auto p = pBegin;
        size_t nextI = *p;
        while (p != pEnd - 1) {
            size_t i = nextI;
            ++p;
            nextI = *p;
            F w = weights_[i];
            F y = outData_[i];
            leftSumW += w;
            rightSumW -= w;
            leftSumWY += w * y;
            rightSumWY -= w * y;
            F score = leftSumWY * leftSumWY / leftSumW + rightSumWY * rightSumWY / rightSumW;

            if (score <= bestScore) continue;  // usually true
            ++n;

            if (leftSumW < minNodeWeight 
                || rightSumW < minNodeWeight
                || static_cast<size_t>(p - pBegin) < options_->minNodeSize() 
                || static_cast<size_t>(pEnd - p) < options_->minNodeSize()
            ) continue;

            float leftX = inData_(i, j);
            float rightX = inData_(nextI, j);
            if (leftX == rightX) continue;      // stricter check?

            bestScore = score;
            bestJ = j;
            bestX = (leftX + rightX) / 2;
            bestLeftY = leftSumWY / leftSumW;
            bestRightY = rightSumWY / rightSumW;
        }

        //{
        //    F w = weights_[nextI];
        //    F y = outData_[nextI];
        //    leftSumW += w;
        //    rightSumW -= w;
        //    leftSumWY += w * y;
        //    rightSumWY -= w * y;
        //    cout << weights_.minCoeff() << " - " << weights_.maxCoeff() << ": " << rightSumW << endl;
        //}

        t = clockCycleCount();
        t2 += t;
        t0 -= t;
    }

    t = clockCycleCount();
    t0 += t;

    if (options_->profile()) {
        cout << t0 << endl;
        cout << t1 << " (" << static_cast<float>(t1) / (sampleCount_ * usedVariableCount) << ")" << endl;
        cout << t2 << " (" << static_cast<float>(t2) / (usedSampleCount * usedVariableCount) << ")" << endl;
        cout << 100.0f * n / ((usedSampleCount - 1) * usedVariableCount) << "%" << endl;
        cout << endl;
    }

    return new StumpPredictor{ variableCount_, bestJ, bestX, static_cast<float>(bestLeftY), static_cast<float>(bestRightY) };
};

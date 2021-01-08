#include "pch.h"
#include "StubTrainer.h"
#include "StubOptions.h"
#include "Algorithms.h"
#include "ClockCycleCount.h"
#include "RNE.h"

StubTrainer::StubTrainer() :
    options_(std::make_unique<StubOptions>())
{}

void StubTrainer::setInData(Eigen::Ref<ArrayXXf> inData)
{ 
    ASSERT(inData.isFinite().all());

    // There should be a better way...
    inData_.~Map();
    new (&inData_) Eigen::Map<ArrayXXf>(&inData(0, 0), inData.rows(), inData.cols());
   
    sampleCount_ = static_cast<int>(inData.rows());
    variableCount_ = static_cast<int>(inData.cols());

    sortedSamples_.resize(variableCount_);
    vector<pair<float, int>> tmp(sampleCount_);
    for (int j = 0; j < variableCount_; ++j) {
        for (int i = 0; i < sampleCount_; ++i)
            tmp[i] = { inData(i,j), i };
        std::sort(begin(tmp), end(tmp));
        sortedSamples_[j].resize(sampleCount_);
        for (int i = 0; i < sampleCount_; ++i)
            sortedSamples_[j][i] = tmp[i].second;
    }
}

void StubTrainer::setOutData(const ArrayXf& outData)
{
    ASSERT(outData.isFinite().all());
    outData_ = outData;
}

void StubTrainer::setWeights(const ArrayXf& weights)
{
    ASSERT(weights.isFinite().all());
    ASSERT((weights > 0).all());
    weights_ = weights;
}

void StubTrainer::setOptions(const AbstractOptions& opt)
{
    const StubOptions& opt1 = dynamic_cast<const StubOptions&>(opt);
    options_.reset(opt1.clone());
}

StubPredictor* StubTrainer::train() const
{
    ASSERT(sampleCount_ > 0);
    ASSERT(variableCount_ > 0);
    ASSERT(outData_.size() == sampleCount_);
    ASSERT(weights_.size() == sampleCount_);

    if (options_->highPrecision())
        return trainImpl_<double>();
    else
        return trainImpl_<float>();
}

template<typename F>
StubPredictor* StubTrainer::trainImpl_() const
{
    cout << sizeof(F) << endl;
    int n = 0;
    int64_t t0 = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;

    int64_t t = clockCycleCount();
    t0 -= t;

    int usedSampleCount = std::max(1, static_cast<int>(options_->usedSampleRatio() * sampleCount_ + 0.5));
    int usedVariableCount = std::max(1, static_cast<int>(options_->usedVariableRatio() * variableCount_ + 0.5));

    // used samples mask

    usedSampleMask_.resize(sampleCount_);
    randomMask(
        begin(usedSampleMask_), 
        end(usedSampleMask_), 
        usedSampleCount, 
        theRNE
    );
 
    // used variables

    usedVariables_.resize(variableCount_);
    std::iota(begin(usedVariables_), end(usedVariables_), 0);
    orderedRandomSubset(
        begin(usedVariables_), 
        end(usedVariables_), 
        begin(usedVariables_), 
        begin(usedVariables_) + usedVariableCount, 
        theRNE
    );
 
    usedVariables_.resize(usedVariableCount);

    // sums

    F sumW = F(0);
    F sumWY = F(0);
    for (int i = 0; i < sampleCount_; ++i) {
        F m = usedSampleMask_[i];
        F w = weights_[i];
        F y = outData_[i];
        sumW += m * w;
        sumWY += m * w * y;
    }

    // find best split

    F bestScore = sumWY * sumWY / sumW;
    int bestJ = 0;
    float bestX = -std::numeric_limits<float>::infinity();
    F bestLeftY = std::numeric_limits<F>::quiet_NaN();
    F bestRightY = sumWY / sumW;

    sortedUsedSamples_.resize(usedSampleCount);

    for (int j : usedVariables_) {

        // prepare list of samples
        t = clockCycleCount();
        t0 += t;
        t1 -= t;

        copyIf(
            begin(sortedSamples_[j]),
            begin(sortedUsedSamples_),
            end(sortedUsedSamples_),
            [&](int i) { return usedSampleMask_[i]; }
        );

        // find best split
        t = clockCycleCount();
        t1 += t;
        t2 -= t;
        
        F leftSumW = F(0);
        F leftSumWY = F(0);
        F rightSumW = sumW;
        F rightSumWY = sumWY;

        auto p = begin(sortedUsedSamples_);
        auto pEnd = end(sortedUsedSamples_);
        int nextI = *p;
        ++p;
        for (; p != pEnd; ++p) {
            int i = nextI;
            nextI = *p;
            F w = weights_[i];
            F y = outData_[i];
            leftSumW += w;
            rightSumW -= w;
            leftSumWY += w * y;
            rightSumWY -= w * y;
            F score = leftSumWY * leftSumWY / leftSumW + rightSumWY * rightSumWY / rightSumW;

            if (score <= bestScore) continue;
            ++n;

            float leftX = inData_(i, j);
            float rightX = inData_(nextI, j);
            if (leftX == rightX) continue;      // stricter check?

            bestScore = score;
            bestJ = j;
            bestX = (leftX + rightX) / 2;
            bestLeftY = leftSumWY / leftSumW;
            bestRightY = rightSumWY / rightSumW;
        }

        t = clockCycleCount();
        t2 += t;
        t0 -= t;

        int i = nextI;
        F w = weights_[i];
        F y = outData_[i];
        leftSumW += w;
        rightSumW -= w;
        leftSumWY += w * y;
        rightSumWY -= w * y;
        ASSERT(rightSumW == 0 && rightSumWY == 0);  // this test only works with integer out data and weights
    }

    t = clockCycleCount();
    t0 += t;

    if (options_->profile()) {
        cout << static_cast<float>(t0) << endl;
        cout << static_cast<float>(t1) << " (" << static_cast<float>(t1) / (sampleCount_ * usedVariableCount) << ")" << endl;
        cout << static_cast<float>(t2) << " (" << static_cast<float>(t2) / (usedSampleCount * usedVariableCount) << ")" << endl;
        cout << 100.0f * n / ((usedSampleCount - 1) * usedVariableCount) << "%" << endl;
        cout << endl;
    }

    return new StubPredictor(variableCount_, bestJ, bestX, static_cast<float>(bestLeftY), static_cast<float>(bestRightY));
};

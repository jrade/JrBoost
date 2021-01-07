#include "pch.h"
#include "StubTrainer.h"
#include "StubOptions.h"
#include "RNE.h"
#include "ClockCycleCount.h"
#include "Algorithms.h"

void StubTrainer::setOptions(const AbstractOptions& opt)
{
    options_.reset(dynamic_cast<const StubOptions&>(opt).clone());
}

void StubTrainer::setInData(const ArrayXXf& inData)
{ 
    // TO DO: check size > 0 and fits in int
        
    inData_ = inData;    // keep reference instead
    sampleCount_ = static_cast<int>(inData.rows());
    variableCount_ = static_cast<int>(inData.cols());

    sortedSamples_.resize(variableCount_);
    vector<pair<float, int>> tmp(sampleCount_);
    for (int j = 0; j < variableCount_; ++j) {
        for (int i = 0; i < sampleCount_; ++i)
            tmp[i] = { inData_(i,j), i };
        std::sort(begin(tmp), end(tmp));
        sortedSamples_[j].resize(sampleCount_);
        for (int i = 0; i < sampleCount_; ++i)
            sortedSamples_[j][i] = tmp[i].second;
    }
}

StubPredictor* StubTrainer::train() const
{
    using acc_t = float;

    int64_t t0 = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t t3 = 0;

    int64_t t = clockCycleCount();
    t0 -= t;

    // TO DO: check that sie of outData_ and weights_ is correct

    int usedSampleCount = std::max(1, static_cast<int>(options_->usedSampleRatio * sampleCount_ + 0.5));
    int usedVariableCount = std::max(1, static_cast<int>(options_->usedVariableRatio * variableCount_ + 0.5));

    // used samples

    usedSampleMask_.resize(sampleCount_);
    t = clockCycleCount();
    t0 += t;
    t1 -= t;
    randomMask(
        begin(usedSampleMask_), end(usedSampleMask_), usedSampleCount, theRNE
    );
    t = clockCycleCount();
    t1 += t;
    t0 -= t;

    acc_t sumW = 0.0;
    acc_t sumWY = 0.0;
    for (int i = 0; i < sampleCount_; ++i) {
        acc_t m = usedSampleMask_[i];
        acc_t w = weights_[i];
        acc_t y = outData_[i];
        sumW += m * w;
        sumWY += m * w * y;
    }

    sortedUsedSamples_.resize(usedSampleCount);

    // used variables

    usedVariables_.resize(variableCount_);
    std::iota(begin(usedVariables_), end(usedVariables_), 0);
    t = clockCycleCount();
    t0 += t;
    t1 -= t;
    orderedRandomSubset(
        begin(usedVariables_), end(usedVariables_), 
        begin(usedVariables_), begin(usedVariables_) + usedVariableCount, 
        theRNE
    );
    t = clockCycleCount();
    t1 += t;
    t0 -= t;

    usedVariables_.resize(usedVariableCount);

    // find best split

    acc_t bestScore = -1.0f;
    int bestJ = -1;
    float bestX = std::numeric_limits<float>::quiet_NaN();
    float bestLeftY = std::numeric_limits<float>::quiet_NaN();
    float bestRightY = std::numeric_limits<float>::quiet_NaN();

    for (int j : usedVariables_) {

        // prepare list of samples

        t = clockCycleCount();
        t0 += t;
        t2 -= t;

        auto p = begin(sortedSamples_[j]);
        auto pEnd = end(sortedSamples_[j]);
        auto q = begin(sortedUsedSamples_);
        auto qEnd = end(sortedUsedSamples_);
        while (p != pEnd) {
            int i = *p;
            *q = i;
            ++p;
            q += usedSampleMask_[i];
        }
        assert(q == qEnd);

        t = clockCycleCount();
        t2 += t;
        t3 -= t;

        // find best split

        acc_t leftSumW = 0.0f;
        acc_t leftSumWY = 0.0f;
        acc_t rightSumW = sumW;
        acc_t rightSumWY = sumWY;

        p = begin(sortedUsedSamples_);
        pEnd = end(sortedUsedSamples_);
        int nextI = *p;
        ++p;

        for (; p != pEnd; ++p) {
            int i = nextI;
            nextI = *p;
            acc_t w = weights_[i];
            acc_t y = outData_[i];
            leftSumW += w;
            rightSumW -= w;
            leftSumWY += w * y;
            rightSumWY -= w * y;
            acc_t score = leftSumWY * leftSumWY / leftSumW + rightSumWY * rightSumWY / rightSumW;
            if (score <= bestScore) continue;

            float leftX = inData_(i, j);
            float rightX = inData_(nextI, j);
            if (leftX == rightX) continue;      // stricter check?

            bestScore = score;
            bestJ = j;
            bestX = (leftX + rightX) / 2;
            bestLeftY = static_cast<float>(leftSumWY / leftSumW);
            bestRightY =static_cast<float>(rightSumWY / rightSumW);
        }

        t = clockCycleCount();
        t3 += t;
        t0 -= t;

        int i = nextI;
        acc_t w = weights_[i];
        acc_t y = outData_[i];
        leftSumW += w;
        rightSumW -= w;
        leftSumWY += w * y;
        rightSumWY -= w * y;
        //cout << rightSumW << " " << rightSumWY << endl;
        //if (rightSumW != 0.0 || rightSumWY != 0.0)   // this test only works with integer outData and weights
        //    throw std::runtime_error("Bad sums");
    }

    t = clockCycleCount();
    t0 += t;

    if (bestJ == -1)
        throw std::runtime_error("Failed to find a split.");

    cout << static_cast<float>(t0) << endl;
    cout << static_cast<float>(t1) << endl;
    cout << static_cast<float>(t2) << " (" << static_cast<float>(t2) / (sampleCount_ * usedVariableCount) << ")" << endl;
    cout << static_cast<float>(t3) << " (" << static_cast<float>(t3) / (usedSampleCount * usedVariableCount) << ")" <<endl;

    return new StubPredictor(variableCount_, bestJ, bestX, bestLeftY, bestRightY);
};

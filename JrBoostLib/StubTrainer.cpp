#include "pch.h"
#include "StubTrainer.h"
#include "StubOptions.h"
#include "RNE.h"
#include "ClockCycleCount.h"

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
    int64_t t = clockCycleCount();
   // cout << "A" << endl;

    // TO DO: check that sie of outData_ and weights_ is correct

    int usedSampleCount = std::max(1, static_cast<int>(options_->usedSampleRatio * sampleCount_ + 0.5));
    int usedVariableCount = std::max(1, static_cast<int>(options_->usedVariableRatio * variableCount_ + 0.5));

    // used samples

    usedSampleMask_.resize(sampleCount_);
    std::fill(begin(usedSampleMask_), begin(usedSampleMask_) + usedSampleCount, static_cast<char>(1));
    std::fill(begin(usedSampleMask_) + usedSampleCount, end(usedSampleMask_), static_cast<char>(0));
    std::shuffle(begin(usedSampleMask_), end(usedSampleMask_), theRNE());

    float sumW = 0.0f;
    float sumWY = 0.0f;
    for (int i = 0; i < sampleCount_; ++i) {
        float m = usedSampleMask_[i];
        float w = weights_[i];
        float y = outData_[i];
        sumW += m * w;
        sumWY += m * w * y;
    }

    //cout << sumW << " " << sumWY << endl;

    // used variables

    usedVariables_.resize(variableCount_);
    std::iota(begin(usedVariables_), end(usedVariables_), 0);
    std::shuffle(begin(usedVariables_), end(usedVariables_), theRNE());
    usedVariables_.resize(usedVariableCount);

    // find best split

    float bestScore = sumWY * sumWY / sumW;
    int bestJ = -1;
    float bestX = std::numeric_limits<float>::quiet_NaN();
    float bestLeftY = std::numeric_limits<float>::quiet_NaN();
    float bestRightY = std::numeric_limits<float>::quiet_NaN();

    sortedUsedSamples_.resize(usedSampleCount);
    for (int j : usedVariables_) {

        // prepare list of samples

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

        // find best split

        float leftSumW = 0.0f;
        float leftSumWY = 0.0f;
        float rightSumW = sumW;
        float rightSumWY = sumWY;

        p = begin(sortedUsedSamples_);
        pEnd = end(sortedUsedSamples_);
        int nextI = *p;
        ++p;
        for (; p != pEnd; ++p) {
            int i = nextI;
            nextI = *p;
            float w = weights_[i];
            float y = outData_[i];
            leftSumW += w;
            rightSumW -= w;
            leftSumWY += w * y;
            rightSumWY -= w * y;
            float score = leftSumWY * leftSumWY / leftSumW + rightSumWY * rightSumWY / rightSumW;
            if (score <= bestScore) continue;

            float leftX = inData_(i, j);
            float rightX = inData_(nextI, j);
            if (leftX == rightX) continue;      // stricter check?

            bestScore = score;
            bestJ = j;
            bestX = (leftX + rightX) / 2;
            bestLeftY = leftSumWY / leftSumW;
            bestRightY = rightSumWY / rightSumW;
        }

        int i = nextI;
        float w = weights_[i];
        float y = outData_[i];
        leftSumW += w;
        rightSumW -= w;
        leftSumWY += w * y;
        rightSumWY -= w * y;
        //cout << rightSumW << " " << rightSumWY << endl;
        if (rightSumW != 0.0f || rightSumWY != 0.0f)
            throw 0;
    }

    if (bestJ == -1)
        throw std::runtime_error("Failed to find a split.");

    t = clockCycleCount() - t;
    cout << t / 1.0e6 << endl;

    return new StubPredictor(variableCount_, bestJ, bestX, bestLeftY, bestRightY);
};

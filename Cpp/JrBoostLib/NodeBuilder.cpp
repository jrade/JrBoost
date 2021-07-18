//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "NodeBuilder.h"
#include "TreeOptions.h"
#include "TrivialPredictor.h"


template<typename SampleIndex>
void NodeBuilder<SampleIndex>::reset(CRefXXf inData, CRefXd outData, CRefXd weights, const TreeOptions & options)
{
    inData_.~CRefXXf();
    new(&inData_) CRefXXf(inData);

    outData_.~CRefXd();
    new(&outData_) CRefXd(outData);

    weights_.~CRefXd();
    new(&weights_) CRefXd(weights);

    options_ = &options;

    sumsInit_ = false;
    splitFound_ = false;
    iterationCount_ = 0;
    slowBranchCount_ = 0;
}


template<typename SampleIndex>
void NodeBuilder<SampleIndex>::update(size_t j, const SampleIndex* samplesBegin, const SampleIndex* samplesEnd)
{
    if (!sumsInit_) {
        initSums_(samplesBegin, samplesEnd);
        sumsInit_ = true;
    }

    if (sumW_ == 0) return;

    // optimizations
    double bestScore = bestScore_;
    const double* pWeights = &weights_.coeffRef(0);
    const double* pOutData = &outData_.coeffRef(0);

    const size_t sampleCount = samplesEnd - samplesBegin;
    const size_t minNodeSize = options_->minNodeSize();

    double leftSumW = 0.0;
    double leftSumWY = 0.0;
    double rightSumW = sumW_;
    double rightSumWY = sumWY_;

    const SampleIndex* p = samplesBegin;
    size_t nextI = *p;

    // this is where most execution time is spent ......

    iterationCount_ += sampleCount;
    size_t slowBranchCount = 0;

    while (p != samplesEnd - 1) {

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

    //..................................................

        ++slowBranchCount;

        if (p < samplesBegin + minNodeSize
            || p > samplesEnd - minNodeSize
            || leftSumW < minNodeWeight_
            || rightSumW < minNodeWeight_
            ) continue;

        const float leftX = inData_(i, j);
        const float rightX = inData_(nextI, j);
        const float midX = (leftX + rightX) / 2;

        if (leftX == midX) continue;

        splitFound_ = true;
        bestScore = score;
        bestJ_ = j;
        bestX_ = midX;
        bestLeftY_ = leftSumWY / leftSumW;
        bestRightY_ = rightSumWY / rightSumW;
        bestLeftSampleCount_ = p - samplesBegin;
    }

    bestScore_ = bestScore;

    bestRightSampleCount_ = sampleCount - bestLeftSampleCount_;
    slowBranchCount_ += slowBranchCount;
}


template<typename SampleIndex>
void NodeBuilder<SampleIndex>::initSums_(const SampleIndex* samplesBegin, const SampleIndex* samplesEnd)
{
    const size_t usedSampleCount = samplesEnd - samplesBegin;

    PROFILE::PUSH(PROFILE::SUMS);
    size_t ITEM_COUNT = usedSampleCount;

    sumW_ = 0.0;
    sumWY_ = 0.0;
    for (auto p = samplesBegin; p != samplesEnd; ++p) {
        const size_t i = *p;
        double w = weights_(i);
        double y = outData_(i);
        sumW_ += w;
        sumWY_ += w * y;
    }

    bestScore_ = sumWY_ * sumWY_ / sumW_;
    tol_ = sumW_ * sqrt(static_cast<double>(usedSampleCount)) * numeric_limits<double>::epsilon() / 2;
    minNodeWeight_ = std::max<double>(options_->minNodeWeight(), tol_);

    PROFILE::POP(ITEM_COUNT);
}


template<typename SampleIndex>
void NodeBuilder<SampleIndex>::initNodes(
    TreePredictor::Node** parent, TreePredictor::Node** child, size_t** childSampleCount) const
{
    if (!splitFound_) {
        (*parent)->isLeaf = true;
        (*parent)->y = static_cast<float>((sumW_ == 0.0) ? 0.0 : sumWY_ / sumW_);
    }
    else {
        (*parent)->isLeaf = false;
        (*parent)->j = static_cast<uint32_t>(bestJ_);
        (*parent)->x = bestX_;

        (*parent)->leftChild = *child;
        (*child)->isLeaf = true;
        (*child)->y = static_cast<float>(bestLeftY_);
        ++*child;
        if (childSampleCount) {
            **childSampleCount = bestLeftSampleCount_;
            ++*childSampleCount;
        }

        (*parent)->rightChild = *child;
        (*child)->isLeaf = true;
        (*child)->y = static_cast<float>(bestRightY_);
        ++*child;
        if (childSampleCount) {
            **childSampleCount = bestRightSampleCount_;
            ++*childSampleCount;
        }

        double bestGain = bestScore_ - sumWY_ * sumWY_ / sumW_;
        ASSERT(bestGain > 0);
        (*parent)->gain = static_cast<float>(bestGain);
    }

    ++*parent;
}

//----------------------------------------------------------------------------------------------------------------------

template class NodeBuilder<uint8_t>;
template class NodeBuilder<uint16_t>;
template class NodeBuilder<uint32_t>;
template class NodeBuilder<uint64_t>;

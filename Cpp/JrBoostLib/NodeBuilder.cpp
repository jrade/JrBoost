//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "NodeBuilder.h"
#include "StumpOptions.h"
#include "TrivialPredictor.h"


template<typename SampleIndex>
void NodeBuilder<SampleIndex>::reset()
{
    sumsInit_ = false;
    splitFound_ = false;
    iterationCount_ = 0;
    slowBranchCount_ = 0;
}


template<typename SampleIndex>
void NodeBuilder<SampleIndex>::update(
    size_t j, CRefXXf inData, CRefXd outData, CRefXd weights, const StumpOptions& options)
{
    const size_t usedSampleCount = sortedSamples_.size();
    const size_t minNodeSize = options.minNodeSize();

    if (!sumsInit_) {
        initSums_(outData, weights, options);
        sumsInit_ = true;
    }

    if (sumW_ == 0) return;


    // find best split .....................................................

    double leftSumW = 0.0;
    double leftSumWY = 0.0;
    double rightSumW = sumW_;
    double rightSumWY = sumWY_;

    const auto pBegin = cbegin(sortedSamples_);
    const auto pEnd = cend(sortedSamples_);
    auto p = pBegin;
    size_t nextI = *p;

    // this is where most execution time is spent ......

    iterationCount_ += usedSampleCount;

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

        if (score <= bestScore_) continue;  // usually true

    //..................................................

        ++slowBranchCount_;

        if (p < pBegin + minNodeSize
            || p > pEnd - minNodeSize
            || leftSumW < minNodeWeight_
            || rightSumW < minNodeWeight_
            ) continue;

        const float leftX = inData(i, j);
        const float rightX = inData(nextI, j);
        const float midX = (leftX + rightX) / 2;

        if (leftX == midX) continue;

        splitFound_ = true;
        bestScore_ = score;
        bestJ_ = j;
        bestX_ = midX;
        bestLeftY_ = leftSumWY / leftSumW;
        bestRightY_ = rightSumWY / rightSumW;
    }
}


template<typename SampleIndex>
void NodeBuilder<SampleIndex>::initSums_(const CRefXd& outData, const CRefXd& weights, const StumpOptions& options)
{
    const size_t usedSampleCount = sortedSamples_.size();

    PROFILE::PUSH(PROFILE::SUMS);
    size_t ITEM_COUNT = usedSampleCount;

    sumW_ = 0.0;
    sumWY_ = 0.0;
    for (size_t i : sortedSamples_) {
        double w = weights(i);
        double y = outData(i);
        sumW_ += w;
        sumWY_ += w * y;
    }

    bestScore_ = sumWY_ * sumWY_ / sumW_;
    tol_ = sumW_ * sqrt(static_cast<double>(usedSampleCount)) * numeric_limits<double>::epsilon() / 2;
    minNodeWeight_ = std::max<double>(options.minNodeWeight(), tol_);

    PROFILE::POP(ITEM_COUNT);
}


template<typename SampleIndex>
void NodeBuilder<SampleIndex>::initNodes(TreePredictor::Node** parent, TreePredictor::Node** child) const
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

        (*parent)->rightChild = *child;
        (*child)->isLeaf = true;
        (*child)->y = static_cast<float>(bestRightY_);
        ++*child;
    }

    ++*parent;
}

//----------------------------------------------------------------------------------------------------------------------

template class NodeBuilder<uint8_t>;
template class NodeBuilder<uint16_t>;
template class NodeBuilder<uint32_t>;
template class NodeBuilder<uint64_t>;

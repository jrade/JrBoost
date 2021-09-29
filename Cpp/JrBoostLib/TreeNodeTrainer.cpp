//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeNodeTrainer.h"

#include "TreeNodeExt.h"
#include "TreeOptions.h"


template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::init(const TreeNodeExt* node, const TreeOptions& options)
{
    sumW_ = node->sumW;
    sumWY_ = node->sumWY;
    minNodeWeight_ = std::max(options.minNodeWeight(), 1e-6 * sumW_);

    splitFound_ = false;
    score_ = sumWY_ * sumWY_ / sumW_ + options.minGain();

    iterationCount_ = 0;
    slowBranchCount_ = 0;
}


template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::update(
    CRefXXf inData,
    CRefXd outData,
    CRefXd weights,
    const TreeOptions& options,
    const SampleIndex* sortedSamplesBegin,
    const SampleIndex* sortedSamplesEnd,
    size_t j
)
{
    const float* pInDataColJ = std::data(inData.col(j));
    const double* pOutData = std::data(outData);
    const double* pWeights = std::data(weights);

    if (sumW_ == 0) return;

    const size_t minNodeSize = options.minNodeSize();

    double leftSumW = 0.0;
    double leftSumWY = 0.0;

    const SampleIndex* p = sortedSamplesBegin;
    size_t nextI = *p;
    while (p != sortedSamplesEnd - 1) {

        // this is where most execution time is spent ..........................

        const size_t i = nextI;
        nextI = *++p;

        const double w = pWeights[i];
        const double y = pOutData[i];
        leftSumW += w;
        leftSumWY += w * y;
        const double rightSumW = sumW_ - leftSumW;
        const double rightSumWY = sumWY_ - leftSumWY;

        const double score = leftSumWY * leftSumWY / leftSumW + rightSumWY * rightSumWY / rightSumW;

        if (score <= score_) continue;  // usually true .......................

        ++slowBranchCount_;

        if (p < sortedSamplesBegin + minNodeSize
            || p > sortedSamplesEnd - minNodeSize
            || leftSumW < minNodeWeight_
            || rightSumW < minNodeWeight_
            ) continue;

        const float leftX = pInDataColJ[i];
        const float rightX = pInDataColJ[nextI];
        const float midX = (leftX + rightX) / 2;

        if (leftX == midX) continue;

        splitFound_ = true;
        score_ = score;
        j_ = j;
        x_ = midX;
        leftSumW_ = leftSumW;
        leftSumWY_ = leftSumWY;
        rightSumW_ = rightSumW;
        rightSumWY_ = rightSumWY;
    }

    iterationCount_ += sortedSamplesBegin - sortedSamplesEnd;
}


template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::join(const TreeNodeTrainer& other)
{
    iterationCount_ += other.iterationCount_;
    slowBranchCount_ += other.slowBranchCount_;

    if (!other.splitFound_) return;
    if (splitFound_ && other.score_ < score_) return;
    if (splitFound_ && other.score_ == score_ && j_ < other.j_) return;     // makes the code deterministic

    splitFound_ = true;
    score_ = other.score_;
    j_ = other.j_;
    x_ = other.x_;
    leftSumW_ = other.leftSumW_;
    leftSumWY_ = other.leftSumWY_;
    rightSumW_ = other.rightSumW_;
    rightSumWY_ = other.rightSumWY_;
}


template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::finalize(TreeNodeExt** parentNode, TreeNodeExt** childNode) const
{
    if (splitFound_) {

        (*parentNode)->isLeaf = false;
        (*parentNode)->j = j_;
        (*parentNode)->x = x_;
        (*parentNode)->gain = static_cast<float>(score_ - sumWY_ * sumWY_ / sumW_);

        (*parentNode)->leftChild = *childNode;
        (*childNode)->isLeaf = true;
        (*childNode)->y = static_cast<float>(leftSumWY_ / leftSumW_);
        ++*childNode;

        (*parentNode)->rightChild = *childNode;
        (*childNode)->isLeaf = true;
        (*childNode)->y = static_cast<float>(rightSumWY_ / rightSumW_);
        ++*childNode;
    }

    ++*parentNode;

    PROFILE::UPDATE_BRANCH_STATISTICS(iterationCount_, slowBranchCount_);
}

//......................................................................................................................

template class TreeNodeTrainer<uint8_t>;
template class TreeNodeTrainer<uint16_t>;
template class TreeNodeTrainer<uint32_t>;
template class TreeNodeTrainer<uint64_t>;

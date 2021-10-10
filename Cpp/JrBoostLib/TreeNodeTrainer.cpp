//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeNodeTrainer.h"

#include "TreeNodeExt.h"
#include "TreeOptions.h"


template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::init(const TreeNodeExt& node, const TreeOptions& options)
{
    sampleCount_ = node.sampleCount;
    sumW_ = node.sumW;
    sumWY_ = node.sumWY;
    minNodeWeight_ = std::max(options.minNodeWeight(), 1e-6 * sumW_);

    splitFound_ = false;
    score_ = sumWY_ * sumWY_ / sumW_ + options.minGain();

    iterationCount_ = 0;
    slowBranchCount_ = 0;
}

template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::init(const TreeNodeTrainer& other)
{
    ASSERT(!other.splitFound_);

    sampleCount_ = other.sampleCount_;
    sumW_ = other.sumW_;
    sumWY_ = other.sumWY_;
    minNodeWeight_ = other.minNodeWeight_;

    splitFound_ = false;
    score_ = other.score_;

    iterationCount_ = 0;
    slowBranchCount_ = 0;
}


template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::update(
    CRefXXfc inData,
    CRefXd outData,
    CRefXd weights,
    const TreeOptions& options,
    const SampleIndex* pSortedSamplesBegin,
    const SampleIndex* pSortedSamplesEnd,
    size_t j
)
{
    ASSERT(static_cast<size_t>(pSortedSamplesEnd - pSortedSamplesBegin) == sampleCount_);

    const float* pInDataColJ = std::data(inData.col(j));
    const double* pOutData = std::data(outData);
    const double* pWeights = std::data(weights);

    if (sumW_ == 0) return;

    const size_t minNodeSize = options.minNodeSize();

    double leftSumW = 0.0;
    double leftSumWY = 0.0;

    const SampleIndex* p = pSortedSamplesBegin;
    size_t nextI = *p;
    while (p != pSortedSamplesEnd - 1) {

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

        if (p < pSortedSamplesBegin + minNodeSize
            || p > pSortedSamplesEnd - minNodeSize
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

    iterationCount_ += sampleCount_;
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
void TreeNodeTrainer<SampleIndex>::finalize(TreeNodeExt** ppParentNode, TreeNodeExt** ppChildNode) const
{
    if (splitFound_) {

        (*ppParentNode)->isLeaf = false;
        (*ppParentNode)->j = j_;
        (*ppParentNode)->x = x_;
        (*ppParentNode)->gain = static_cast<float>(score_ - sumWY_ * sumWY_ / sumW_);

        (*ppParentNode)->leftChild = *ppChildNode;
        (*ppChildNode)->isLeaf = true;
        (*ppChildNode)->y = static_cast<float>(leftSumWY_ / leftSumW_);
        ++*ppChildNode;

        (*ppParentNode)->rightChild = *ppChildNode;
        (*ppChildNode)->isLeaf = true;
        (*ppChildNode)->y = static_cast<float>(rightSumWY_ / rightSumW_);
        ++*ppChildNode;
    }

    ++*ppParentNode;

    PROFILE::UPDATE_BRANCH_STATISTICS(iterationCount_, slowBranchCount_);
}

//......................................................................................................................

template class TreeNodeTrainer<uint8_t>;
template class TreeNodeTrainer<uint16_t>;
template class TreeNodeTrainer<uint32_t>;
template class TreeNodeTrainer<uint64_t>;

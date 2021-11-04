//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeNodeTrainer.h"

#if PACKED_DATA
#include "TreeTrainerBase.h"
#endif

#include "BaseOptions.h"


template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::init(const TreeNodeExt& node, const BaseOptions& options)
{
    sampleCount_ = node.sampleCount;
    sumW_ = node.sumW;
    sumWY_ = node.sumWY;
    minNodeWeight_ = std::max(options.minNodeWeight(), 1e-6 * sumW_);
    minNodeSize_ = options.minNodeSize();

    splitFound_ = false;
    score_ = square(sumWY_) / sumW_ + options.minNodeGain();

    iterationCount_ = 0;
    slowBranchCount_ = 0;
}


template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::fork(TreeNodeTrainer* other) const
{
    ASSERT(!splitFound_);

    other->sampleCount_ = sampleCount_;
    other->sumW_ = sumW_;
    other->sumWY_ = sumWY_;
    other->minNodeWeight_ = minNodeWeight_;
    other->minNodeSize_ = minNodeSize_;

    other->splitFound_ = false;
    other->score_ = score_;

    other->iterationCount_ = 0;
    other->slowBranchCount_ = 0;
}


// finds the best split of a node for variable j

template<typename SampleIndex>
void TreeNodeTrainer<SampleIndex>::update(
    CRefXXfc inData,
#if PACKED_DATA
    const WyPack* pWyPacks,
#else
    CRefXd outData,
    CRefXd weights,
#endif
    const SampleIndex* pSortedSamplesBegin,
    const SampleIndex* pSortedSamplesEnd,
    size_t j
)
{
    // the samples in the range [pSortedSamplesBegin, pSortedSamplesEnd) should be sorted according to inData.col(j)

    ASSERT(static_cast<size_t>(pSortedSamplesEnd - pSortedSamplesBegin) == sampleCount_);

    if (sumW_ == 0) return;

    const float* pInDataColJ = std::data(inData.col(j));

#if !PACKED_DATA
    const double* pOutData = std::data(outData);
    const double* pWeights = std::data(weights);
#endif
    double leftSumW = 0.0;
    double leftSumWY = 0.0;

    const SampleIndex* p = pSortedSamplesBegin;
    size_t nextI = *p;
    while (p != pSortedSamplesEnd - 1) {

        // this is where most execution time is spent ..........................

        const size_t i = nextI;
        nextI = *++p;

#if PACKED_DATA
        const double w = pWyPacks[i].w;
        const double wy = pWyPacks[i].wy;
        leftSumW += w;
        leftSumWY += wy;
#else
        const double w = pWeights[i];
        const double y = pOutData[i];
        leftSumW += w;
        leftSumWY += w * y;
#endif
        const double rightSumW = sumW_ - leftSumW;
        const double rightSumWY = sumWY_ - leftSumWY;

        const double score = square(leftSumWY) / leftSumW + square(rightSumWY) / rightSumW;

        if (score <= score_) continue;  // usually true .......................

        ++slowBranchCount_;

        if (p < pSortedSamplesBegin + minNodeSize_
            || p > pSortedSamplesEnd - minNodeSize_
            || leftSumW < minNodeWeight_
            || rightSumW < minNodeWeight_
        ) continue;

        const float leftX = pInDataColJ[i];
        const float rightX = pInDataColJ[nextI];     // leftX <= rightX
        const float midX = (leftX + rightX) / 2;

        if (leftX == midX) continue;

        splitFound_ = true;
        score_ = score;
        j_ = j;
        x_ = midX;
        leftSampleCount_ = p - pSortedSamplesBegin;
        leftSumW_ = leftSumW;
        leftSumWY_ = leftSumWY;
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
    if (splitFound_ && other.score_ == score_ && j_ < other.j_) return;     // makes joining deterministic

    splitFound_ = true;
    score_ = other.score_;
    j_ = other.j_;
    x_ = other.x_;
    leftSampleCount_ = other.leftSampleCount_;
    leftSumW_ = other.leftSumW_;
    leftSumWY_ = other.leftSumWY_;
}


// updates one node based on the best split found
// returns the number of used samples in the child nodes if any, otherwise 0

template<typename SampleIndex>
size_t TreeNodeTrainer<SampleIndex>::finalize(TreeNodeExt** ppParentNode, TreeNodeExt** ppChildNode) const
{
    PROFILE::UPDATE_BRANCH_STATISTICS(iterationCount_, slowBranchCount_);

    TreeNodeExt* pParentNode = *ppParentNode;
    ++*ppParentNode;

    if (!splitFound_) return 0;

    pParentNode->isLeaf = false;
    pParentNode->j = j_;
    pParentNode->x = x_;
    pParentNode->gain = static_cast<float>(score_ - square(sumWY_) / sumW_);

    TreeNodeExt* leftChildNode = *ppChildNode;
    ++*ppChildNode;
    pParentNode->leftChild = leftChildNode;

    leftChildNode->isLeaf = true;
    leftChildNode->y = static_cast<float>(leftSumWY_ / leftSumW_);
    leftChildNode->sampleCount = leftSampleCount_;
    leftChildNode->sumW = leftSumW_;
    leftChildNode->sumWY = leftSumWY_;

    TreeNodeExt* rightChildNode = *ppChildNode;
    ++*ppChildNode;
    pParentNode->rightChild = rightChildNode;

    size_t rightSampleCount = sampleCount_ - leftSampleCount_;
    double rightSumW = sumW_ - leftSumW_;
    double rightSumWY = sumWY_ - leftSumWY_;

    rightChildNode->isLeaf = true;
    rightChildNode->y = static_cast<float>(rightSumWY / rightSumW);
    rightChildNode->sampleCount = rightSampleCount;
    rightChildNode->sumW = rightSumW;
    rightChildNode->sumWY = rightSumWY;

    return sampleCount_;
}

//......................................................................................................................

template class TreeNodeTrainer<uint8_t>;
template class TreeNodeTrainer<uint16_t>;
template class TreeNodeTrainer<uint32_t>;
template class TreeNodeTrainer<uint64_t>;

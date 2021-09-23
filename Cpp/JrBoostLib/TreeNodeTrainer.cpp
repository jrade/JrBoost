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
    span<const SampleIndex> sortedSamples,
    size_t j
)
{
    const float* pInDataColJ = &inData.coeffRef(0, j);
    const double* pOutData = &outData.coeffRef(0);
    const double* pWeights = &weights.coeffRef(0);

    if (sumW_ == 0) return;

    const size_t minNodeSize = options.minNodeSize();

    double leftSumW = 0.0;
    double leftSumWY = 0.0;

    auto p = begin(sortedSamples);
    size_t nextI = *p;
    while (p != end(sortedSamples) - 1) {

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

        if (p < begin(sortedSamples) + minNodeSize
            || p > end(sortedSamples) - minNodeSize
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
        leftY_ = leftSumWY / leftSumW;
        rightY_ = rightSumWY / rightSumW;
    }

    iterationCount_ += size(sortedSamples);
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
        (*childNode)->y = static_cast<float>(leftY_);
        ++*childNode;

        (*parentNode)->rightChild = *childNode;
        (*childNode)->isLeaf = true;
        (*childNode)->y = static_cast<float>(rightY_);
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

//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "Tree.h"

class BaseOptions;
struct WyPack;

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
class TreeNodeTrainer {   // immutable class
public:
    TreeNodeTrainer() = default;
    ~TreeNodeTrainer() = default;

    void init(const TreeNodeExt& node, const BaseOptions& options);
    void update(
        CRefXXfc inData, CRefXd outData, CRefXd weights, const SampleIndex* pSortedSamplesBegin,
        const SampleIndex* pSortedSamplesEnd, size_t j);
    size_t finalize(TreeNodeExt** ppParentNode, TreeNodeExt** ppChildNode) const;

    void fork(TreeNodeTrainer* other) const;
    void join(const TreeNodeTrainer& other);

    // not really used, but required by vector<TreeNodeTrainer>
    TreeNodeTrainer(const TreeNodeTrainer&){};
    TreeNodeTrainer& operator=(const TreeNodeTrainer&) { return *this; };

private:
    size_t sampleCount_;
    double sumW_;
    double sumWY_;
    double minNodeWeight_;
    size_t minNodeSize_;

    bool splitFound_;
    double score_;
    size_t j_;
    float x_;

    size_t leftSampleCount_;
    double leftSumW_;
    double leftSumWY_;

    size_t iterationCount_;
    size_t slowBranchCount_;
};

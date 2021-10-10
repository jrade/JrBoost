//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

struct TreeNodeExt;
class TreeOptions;

//----------------------------------------------------------------------------------------------------------------------

template<typename SampleIndex>
class TreeNodeTrainer
{
public:
    TreeNodeTrainer() = default;
    ~TreeNodeTrainer() = default;

    void init(const TreeNodeExt& node, const TreeOptions& options);
    void init(const TreeNodeTrainer& other);
    void update(
        CRefXXfc inData, CRefXd outData, CRefXd weights, const TreeOptions& options,
        const SampleIndex* pSortedSamplesBegin, const SampleIndex* pSortedSamplesEnd, size_t j);
    void join(const TreeNodeTrainer& other);
    void finalize(TreeNodeExt** ppParentNode, TreeNodeExt** ppChildNode) const;

    // not really used, but required by vector<TreeNodeTrainer>
    TreeNodeTrainer(const TreeNodeTrainer&) = default;
    TreeNodeTrainer& operator=(const TreeNodeTrainer&) = default;


private:
    size_t sampleCount_;
    double sumW_;
    double sumWY_;
    double minNodeWeight_;

    bool splitFound_;
    double score_;
    size_t j_;
    float x_;
    double leftSumW_;
    double leftSumWY_;
    double rightSumW_;
    double rightSumWY_;

    size_t iterationCount_;
    size_t slowBranchCount_;
};

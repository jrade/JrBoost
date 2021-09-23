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
    TreeNodeTrainer(TreeNodeTrainer&&) = default;
    TreeNodeTrainer& operator=(TreeNodeTrainer&&) = default;

    void init(const TreeNodeExt* node, const TreeOptions& options);
    void update(CRefXXf inData, CRefXd outData, CRefXd weights,
        const TreeOptions& options, span<const SampleIndex> sortedSamples, size_t j);
    void finalize(TreeNodeExt** parentNode, TreeNodeExt** childNode) const;

// deleted:
    TreeNodeTrainer(const TreeNodeTrainer&) = delete;
    TreeNodeTrainer& operator=(const TreeNodeTrainer&) = delete;

private:
    double sumW_;
    double sumWY_;
    double minNodeWeight_;

    bool splitFound_;
    double score_;
    size_t j_;
    float x_;
    double leftY_;
    double rightY_;

    size_t iterationCount_;
    size_t slowBranchCount_;
};

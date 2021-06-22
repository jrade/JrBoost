// Copyright (C) 2021 Johan Rade <johan.rade@gmail.com>
//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "TreePredictor.h"

class StumpOptions;


template<typename SampleIndex>
class NodeBuilder
{
public:
    NodeBuilder() { reset();  }
    ~NodeBuilder() = default;
    NodeBuilder(NodeBuilder&&) = default;
    NodeBuilder& operator=(NodeBuilder&&) = default;

    vector<SampleIndex>& sortedSamples() { return sortedSamples_; }

    void reset();
    void update(size_t j, CRefXXf inData, CRefXd outData, CRefXd weights, const StumpOptions& options);
    void initNodes(TreePredictor::Node** parent, TreePredictor::Node** child) const;

    size_t iterationCount() const { return iterationCount_; }
    size_t slowBranchCount() const { return slowBranchCount_; }

private:
    void initSums_(const CRefXd& outData, const CRefXd& weights, const StumpOptions& options);

private:
    vector<SampleIndex> sortedSamples_;

    bool sumsInit_;
    double sumW_;
    double sumWY_;
    double tol_;   // estimate of the rounding off error we can expect in rightSumW towards the end of the loop
    double minNodeWeight_;

    bool splitFound_;
    size_t bestJ_;
    float bestX_;
    double bestLeftY_;
    double bestRightY_;

    double bestScore_;

    size_t iterationCount_;
    size_t slowBranchCount_;


// deleted:
    NodeBuilder(const NodeBuilder&) = delete;
    NodeBuilder& operator=(const NodeBuilder&) = delete;
};

// Copyright (C) 2021 Johan Rade <johan.rade@gmail.com>
//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "TreePredictor.h"

class TreeOptions;


template<typename SampleIndex>
class NodeBuilder
{
public:
    // only to keep compiler happy, do not do anything useful, instead rely on reset()
    NodeBuilder() : inData_(dummyXXf_), outData_(dummyXd_), weights_(dummyXd_) {}
    NodeBuilder(const NodeBuilder&) : inData_(dummyXXf_), outData_(dummyXd_), weights_(dummyXd_) {}
    NodeBuilder& operator=(const NodeBuilder&) { return *this; }

    ~NodeBuilder() = default;

    void reset(CRefXXf inData, CRefXd outData, CRefXd weights, const TreeOptions& options);
    void update(size_t j, const SampleIndex* samplesBegin, const SampleIndex* samplesEnd);
    void initNodes(TreePredictor::Node** parent, TreePredictor::Node** child, size_t** childSampleCount) const;

    static size_t iterationCount() { return iterationCount_; }
    static size_t slowBranchCount() { return slowBranchCount_; }

private:
    void initSums_(const SampleIndex* samplesBegin, const SampleIndex* samplesEnd);

private:
    CRefXXf inData_;
    CRefXd outData_;
    CRefXd weights_;
    const TreeOptions* options_;
        
    bool sumsInit_ = false;
    double sumW_;
    double sumWY_;
    double tol_;   // estimate of the rounding off error we can expect in rightSumW towards the end of the loop
    double minNodeWeight_;

    bool splitFound_ = false;
    size_t bestJ_;
    float bestX_;
    double bestLeftY_;
    double bestRightY_;
    size_t bestLeftSampleCount_;
    size_t bestRightSampleCount_;

    double bestScore_;

    inline static thread_local size_t iterationCount_ = 0;
    inline static thread_local size_t slowBranchCount_ = 0;

    inline static ArrayXXf dummyXXf_;
    inline static ArrayXd dummyXd_;
};

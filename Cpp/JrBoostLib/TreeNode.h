// Copyright (C) 2021 Johan Rade <johan.rade@gmail.com>
//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


struct TreeNode
{
    // true for leaf nodes, false for interior nodes
    bool isLeaf;

    // only used by leaf nodes
    // but pruning can turn interior nodes into leaf nodes, so we better set these for all nodes
    float y;

    // only used by interior nodes
    uint32_t j;
    float x;
    float gain;
    TreeNode* leftChild;
    TreeNode* rightChild;
};


size_t count(const TreeNode* node);
size_t depth(const TreeNode* node);
float maxGain(const TreeNode* node);

void prune(TreeNode* node, float pruneLimit);
pair<TreeNode*, vector<TreeNode>> cloneDepthFirst(const TreeNode* sourceNode);
pair<TreeNode*, vector<TreeNode>> cloneBreadthFirst(const TreeNode* sourceNode);

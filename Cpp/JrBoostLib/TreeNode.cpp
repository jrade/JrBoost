//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreeNode.h"


size_t count(const TreeNode* node)
{
    if (node->isLeaf) return 1;
    return 1 + count(node->leftChild) + count(node->rightChild);
}

size_t depth(const TreeNode* node)
{
    if (node->isLeaf) return 0;
    return 1 + std::max(depth(node->leftChild), depth(node->rightChild));
}

float maxGain(const TreeNode* node)
{
    if (node->isLeaf) return 0.0f;
    return std::max(node->gain, std::max(maxGain(node->leftChild), maxGain(node->rightChild)));
}

void prune(TreeNode* node, float pruneLimit)
{
    if (node->isLeaf) return;
    prune(node->leftChild, pruneLimit);
    prune(node->rightChild, pruneLimit);
    if (node->leftChild->isLeaf && node->rightChild->isLeaf && node->gain < pruneLimit)
        node->isLeaf = true;
}

//----------------------------------------------------------------------------------------------------------------------

TreeNode* cloneDepthFirstImpl_(const TreeNode* sourceNode, TreeNode* targetNode)
{
    *targetNode = *sourceNode;
    if (targetNode->isLeaf) return targetNode + 1;
    targetNode->leftChild = targetNode + 1;
    targetNode->rightChild = cloneDepthFirstImpl_(sourceNode->leftChild, targetNode->leftChild);
    return cloneDepthFirstImpl_(sourceNode->rightChild, targetNode->rightChild);
}

pair<TreeNode*, vector<TreeNode>> cloneDepthFirst(const TreeNode* sourceNode)
{
    size_t n = count(sourceNode);
    vector<TreeNode> targetNodes(n);
    TreeNode* targetNode = data(targetNodes);
    cloneDepthFirstImpl_(sourceNode, targetNode);
    return std::make_pair(targetNode, move(targetNodes));
}

pair<TreeNode*, vector<TreeNode>> cloneBreadthFirst(const TreeNode* sourceNode)
{
    size_t n = count(sourceNode);
    vector<TreeNode> targetNodes(n);

    TreeNode* targetParentNode = data(targetNodes);
    TreeNode* targetChildNode  = data(targetNodes) + 1;
    TreeNode* targetNodeEnd    = data(targetNodes) + n;

    *targetParentNode = *sourceNode;

    for (; targetParentNode != targetNodeEnd; ++targetParentNode) {

        if (targetParentNode->isLeaf) continue;

        *targetChildNode = *(targetParentNode->leftChild);
        targetParentNode->leftChild = targetChildNode;
        ++targetChildNode;

        *targetChildNode = *(targetParentNode->rightChild);
        targetParentNode->rightChild = targetChildNode;
        ++targetChildNode;
    }

    return std::make_pair(data(targetNodes), move(targetNodes));
}

//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "Tree.h"

#include "Base128Encoding.h"


namespace TreeTools {

size_t nodeCount(const TreeNode* node)
{
    if (node->isLeaf)
        return 1;
    return 1 + nodeCount(node->leftChild) + nodeCount(node->rightChild);
}

size_t treeDepth(const TreeNode* node)
{
    if (node->isLeaf)
        return 0;
    return 1 + std::max(treeDepth(node->leftChild), treeDepth(node->rightChild));
}

float maxNodeGain(const TreeNode* node)
{
    if (node->isLeaf)
        return 0.0f;
    return std::max({node->gain, maxNodeGain(node->leftChild), maxNodeGain(node->rightChild)});
}


void pruneTree(TreeNode* node, float minNodeGain)
{
    if (node->isLeaf)
        return;
    pruneTree(node->leftChild, minNodeGain);
    pruneTree(node->rightChild, minNodeGain);
    if (node->leftChild->isLeaf && node->rightChild->isLeaf && node->gain < minNodeGain)
        node->isLeaf = true;
}

TreeNode* cloneTreeDepthFirstImpl_(const TreeNode* sourceNode, TreeNode* targetNode)
{
    *targetNode = *sourceNode;
    if (targetNode->isLeaf)
        return targetNode + 1;
    targetNode->leftChild = targetNode + 1;
    targetNode->rightChild = cloneTreeDepthFirstImpl_(sourceNode->leftChild, targetNode->leftChild);
    return cloneTreeDepthFirstImpl_(sourceNode->rightChild, targetNode->rightChild);
}

vector<TreeNode> cloneTreeDepthFirst(const TreeNode* sourceRoot)
{
    const size_t n = nodeCount(sourceRoot);
    vector<TreeNode> targetNodes(n);
    TreeNode* targetRoot = data(targetNodes);
    cloneTreeDepthFirstImpl_(sourceRoot, targetRoot);
    return targetNodes;
}

vector<TreeNode> cloneTreeBreadthFirst(const TreeNode* sourceNode)
{
    size_t n = nodeCount(sourceNode);
    vector<TreeNode> targetNodes(n);

    TreeNode* targetParentNode = data(targetNodes);
    TreeNode* targetChildNode = data(targetNodes) + 1;
    TreeNode* targetNodeEnd = data(targetNodes) + n;

    *targetParentNode = *sourceNode;

    for (; targetParentNode != targetNodeEnd; ++targetParentNode) {

        if (targetParentNode->isLeaf)
            continue;

        *targetChildNode = *(targetParentNode->leftChild);
        targetParentNode->leftChild = targetChildNode;
        ++targetChildNode;

        *targetChildNode = *(targetParentNode->rightChild);
        targetParentNode->rightChild = targetChildNode;
        ++targetChildNode;
    }

    return targetNodes;
}

TreeNode* reindexTreeImpl_(const TreeNode* sourceNode, TreeNode* targetNode, CRefXs newIndices)
{
    *targetNode = *sourceNode;
    if (targetNode->isLeaf)
        return targetNode + 1;
    targetNode->j = newIndices(targetNode->j);
    targetNode->leftChild = targetNode + 1;
    targetNode->rightChild = reindexTreeImpl_(sourceNode->leftChild, targetNode->leftChild, newIndices);
    return reindexTreeImpl_(sourceNode->rightChild, targetNode->rightChild, newIndices);
}

vector<TreeNode> reindexTree(const TreeNode* sourceRoot, CRefXs newIndices)
{
    const size_t n = nodeCount(sourceRoot);
    vector<TreeNode> targetNodes(n);
    TreeNode* targetRoot = data(targetNodes);
    reindexTreeImpl_(sourceRoot, targetRoot, newIndices);
    return targetNodes;
}


void predict(const TreeNode* root, CRefXXfc inData, double c, RefXd outData)
{
    const size_t sampleCount = inData.rows();
    for (size_t i = 0; i != sampleCount; ++i) {
        const TreeNode* node = root;
        while (!node->isLeaf)
            node = (inData(i, node->j) < node->x) ? node->leftChild : node->rightChild;
        outData(i) += c * node->y;
    }
}

double predictOne(const TreeNode* node, CRefXf inData)
{
    while (!node->isLeaf)
        node = (inData(node->j) < node->x) ? node->leftChild : node->rightChild;
    return node->y;
}

size_t variableCount(const TreeNode* node)
{
    if (node->isLeaf)
        return 0;
    return std::max({node->j + 1, variableCount(node->leftChild), variableCount(node->rightChild)});
}

void variableWeights(const TreeNode* node, double c, RefXd weights)
{
    if (node->isLeaf)
        return;
    weights(node->j) += c * node->gain;
    variableWeights(node->leftChild, c, weights);
    variableWeights(node->rightChild, c, weights);
}

//......................................................................................................................

void saveTreeImpl_(const TreeNode* node, ostream& os)
{
    os.put(static_cast<char>(node->isLeaf));
    if (node->isLeaf)
        os.write(reinterpret_cast<const char*>(&node->y), sizeof(node->y));
    else {
        base128Save(os, node->j);
        os.write(reinterpret_cast<const char*>(&node->x), sizeof(node->x));
        os.write(reinterpret_cast<const char*>(&node->gain), sizeof(node->gain));
        saveTreeImpl_(node->leftChild, os);
        saveTreeImpl_(node->rightChild, os);
    }
}

void saveTree(const TreeNode* root, ostream& os)
{
    const size_t nodeCount = TreeTools::nodeCount(root);
    base128Save(os, nodeCount);
    saveTreeImpl_(root, os);
}


TreeNode* loadTreeImpl_(TreeNode* node, istream& is, int version)
{
    const int isLeaf = is.get();
    if (isLeaf != 0 && isLeaf != 1)
        parseError(is);
    node->isLeaf = static_cast<bool>(isLeaf);

    if (node->isLeaf) {
        is.read(reinterpret_cast<char*>(&node->y), sizeof(node->y));
        return node + 1;
    }
    else {
        node->y = numeric_limits<float>::quiet_NaN();
        if (version >= 5)
            node->j = base128Load(is);
        else {
            uint32_t j32;
            is.read(reinterpret_cast<char*>(&j32), sizeof(j32));
            node->j = static_cast<uint64_t>(j32);
        }

        is.read(reinterpret_cast<char*>(&node->x), sizeof(node->x));

        if (version >= 3 && version < 5)
            is.read(reinterpret_cast<char*>(&node->gain), sizeof(node->gain));
        else if (version < 8)
            node->gain = numeric_limits<float>::quiet_NaN();
        else
            is.read(reinterpret_cast<char*>(&node->gain), sizeof(node->gain));

        node->leftChild = node + 1;
        node->rightChild = loadTreeImpl_(node->leftChild, is, version);
        return loadTreeImpl_(node->rightChild, is, version);
    }
}

vector<TreeNode> loadTree(istream& is, int version)
{
    size_t nodeCount;
    if (version < 5)
        is.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
    else
        nodeCount = base128Load(is);
    vector<TreeNode> nodes = vector<TreeNode>(nodeCount);

    TreeNode* root = data(nodes);
    TreeNode* node = loadTreeImpl_(root, is, version);
    if (node != data(nodes) + size(nodes))
        parseError(is);

    return nodes;
}

}   // namespace TreeTools

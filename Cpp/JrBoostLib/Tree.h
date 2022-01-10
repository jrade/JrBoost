//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


struct TreeNode {
    // true for leaf nodes, false for interior nodes
    bool isLeaf;

    // only used by leaf nodes
    // however, pruning can turn interior nodes into leaf nodes, so y should be set for interior nodes as well
    float y;

    // only used by interior nodes
    size_t j;
    float x;
    float gain;
    TreeNode* leftChild;
    TreeNode* rightChild;
};


namespace TreeTools {

size_t nodeCount(const TreeNode* node);
size_t treeDepth(const TreeNode* node);
float maxNodeGain(const TreeNode* node);

void pruneTree(TreeNode* node, float minNodeGain);
vector<TreeNode> cloneTree(const TreeNode* node);   // first node in the returned vector is the root
vector<TreeNode> reindexTree(const TreeNode* node, CRefXs newIndices);

void predict(const TreeNode* node, CRefXXfc inData, double c, RefXd outData);
double predictOne(const TreeNode* node, CRefXf inData);
size_t variableCount(const TreeNode* node);
void variableWeights(const TreeNode* node, double c, RefXd weights);

void saveTree(const TreeNode* node, ostream& os);
vector<TreeNode> loadTree(istream& is, int version);   // first node in the returned vector is the root

}   // namespace TreeTools

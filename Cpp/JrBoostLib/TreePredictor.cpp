//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreePredictor.h"
#include "TreeNode.h"


void TreePredictor::predict_(CRefXXf inData, double c, RefXd outData) const
{
    const size_t sampleCount = inData.rows();
    for (size_t i = 0; i < sampleCount; ++i) {
        const TreeNode* node = root_;
        while (!node->isLeaf)
            node = (inData(i, node->j) < node->x) ? node->leftChild : node->rightChild;
        outData(i) += c * node->y;
    }
}

//----------------------------------------------------------------------------------------------------------------------

void TreePredictor::save_(ostream& os) const
{
    const int type = Tree;
    os.put(static_cast<char>(type));

    size_t nodeCount = count(root_);
    os.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));

    save_(os, root_);
}

// saves a subtree in depth-first order

void TreePredictor::save_(ostream& os, const TreeNode* node) const
{
    os.put(static_cast<char>(node->isLeaf));
    if (node->isLeaf)
        os.write(reinterpret_cast<const char*>(&node->y), sizeof(node->y));
    else {
        os.write(reinterpret_cast<const char*>(&node->j), sizeof(node->j));
        os.write(reinterpret_cast<const char*>(&node->x), sizeof(node->x));
        os.write(reinterpret_cast<const char*>(&node->gain), sizeof(node->gain));
        save_(os, node->leftChild);
        save_(os, node->rightChild);
    }
}

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<BasePredictor> TreePredictor::load_(istream& is, int version)
{
    size_t nodeCount;
    is.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
    vector<TreeNode> nodes = vector<TreeNode>(nodeCount);

    TreeNode* root = &(nodes.front());
    TreeNode* node = load_(is, root, version);
    if (node != &(nodes.back()) + 1)
        parseError_(is);

    return unique_ptr<TreePredictor>(new TreePredictor(root, nodes));
}


// loads a subtree in depth-first order
// the nodes are stored contiguously with the first node at 'node'
// returns a pointer to one past the last node 

TreeNode* TreePredictor::load_(istream& is, TreeNode* node, int version)
{
    int isLeaf = is.get();
    if (isLeaf != 0 && isLeaf != 1)
        parseError_(is);
    node->isLeaf = static_cast<bool>(isLeaf);

    if (node->isLeaf) {
        is.read(reinterpret_cast<char*>(&node->y), sizeof(node->y));
        return node + 1;
    }
    else {
        is.read(reinterpret_cast<char*>(&node->j), sizeof(node->j));
        is.read(reinterpret_cast<char*>(&node->x), sizeof(node->x));
        if (version >= 3)
            is.read(reinterpret_cast<char*>(&node->gain), sizeof(node->gain));
        node->leftChild = node + 1;
        node->rightChild = load_(is, node->leftChild, version);
        return load_(is, node->rightChild, version);
    }
}

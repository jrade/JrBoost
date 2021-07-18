//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "TreePredictor.h"


void TreePredictor::predict_(CRefXXf inData, double c, RefXd outData) const
{
    const size_t sampleCount = inData.rows();
    for (size_t i = 0; i < sampleCount; ++i) {
        const Node* node = root_;
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

    size_t nodeCount = nodeCount_();
    os.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));

    save_(os, root_);
}

// saves a subtree in depth-first order

void TreePredictor::save_(ostream& os, const Node* node) const
{
    os.put(static_cast<char>(node->isLeaf));
    if (node->isLeaf)
        os.write(reinterpret_cast<const char*>(&node->y), sizeof(node->y));
    else {
        os.write(reinterpret_cast<const char*>(&node->j), sizeof(node->j));
        os.write(reinterpret_cast<const char*>(&node->x), sizeof(node->x));
        save_(os, node->leftChild);
        save_(os, node->rightChild);
    }
}

size_t TreePredictor::nodeCount_() const { return nodeCount_(root_); }

size_t TreePredictor::nodeCount_(const Node* node)
{
    if (node->isLeaf)
        return 1;
    else
        return 1 + nodeCount_(node->leftChild) + nodeCount_(node->rightChild);
}

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<BasePredictor> TreePredictor::load_(istream& is, int /*version*/)
{
    size_t nodeCount;
    is.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
    vector<Node> nodes = vector<Node>(nodeCount);

    Node* root = &(nodes.front());
    Node* node = load_(is, root);
    if (node != &(nodes.back()) + 1)
        parseError_(is);

    return unique_ptr<TreePredictor>(new TreePredictor(root, nodes));
}


// loads a subtree in depth-first order
// the nodes are stored contiguously with the first node at 'node'
// returns a pointer to one past the last node 

TreePredictor::Node* TreePredictor::load_(istream& is, Node* node)
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
        node->leftChild = node + 1;
        node->rightChild = load_(is, node->leftChild);
        return load_(is, node->rightChild);
    }
}

//----------------------------------------------------------------------------------------------------------------------

void TreePredictor::prune(Node* node, float pruneFactor)
{
    if (pruneFactor == 0) return;
    //float pruneLimit = pruneFactor * maxGain_(node);
    float pruneLimit = pruneFactor * node->gain;
    pruneImpl_(node, pruneLimit);
}

float TreePredictor::maxGain_(const Node* node)
{
    if (node->isLeaf) return 0.0f;
    return std::max(node->gain, std::max(maxGain_(node->leftChild), maxGain_(node->rightChild)));
}

void TreePredictor::pruneImpl_(Node* node, float pruneLimit)
{
    if (node->isLeaf) return;
    pruneImpl_(node->leftChild, pruneLimit);
    pruneImpl_(node->rightChild, pruneLimit);
    if (node->leftChild->isLeaf && node->rightChild->isLeaf && node->gain < pruneLimit)
        node->isLeaf = true;
}

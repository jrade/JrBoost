//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BasePredictor.h"


class TreePredictor : public BasePredictor {
public:
    struct Node
    {
        // true for leaf nodes, false for interior nodes
        bool isLeaf;

        // used for leaf nodes
        float y;

        // used for interior nodes
        uint32_t j;
        float x;
        Node* leftChild;
        Node* rightChild;
    };

public:
    template<typename NodeContainer>
    TreePredictor(const Node* root, NodeContainer&& nodeContainer) :
        root_(root),
        nodeContainer_(std::move(nodeContainer))
    {}
 
    virtual ~TreePredictor() = default;

private:
    virtual void predict_(CRefXXf inData, double c, RefXd outData) const;

    virtual void save_(ostream& os) const;
    void save_(ostream& os, const Node* node) const;
    size_t nodeCount_() const;
    static size_t nodeCount_(const Node* node);

    friend class BasePredictor;
    static unique_ptr<BasePredictor> load_(istream& is, int version);
    static Node* load_(istream& is, Node* node);

private:
    const Node* const root_;
    const std::any nodeContainer_;
};

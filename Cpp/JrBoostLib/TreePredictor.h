//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "BasePredictor.h"

struct TreeNode;


class TreePredictor : public BasePredictor
{
public:
    template<typename NodeContainer>
    TreePredictor(const TreeNode* root, NodeContainer&& nodeContainer) :
        root_(root),
        nodeContainer_(std::move(nodeContainer))
    {}
 
    virtual ~TreePredictor() = default;

private:
    virtual void predict_(CRefXXf inData, double c, RefXd outData) const;

    virtual void variableWeights_(double c, RefXd weights) const;
    void variableWeights_(double c, const TreeNode* node, RefXd weights) const;

    virtual void save_(ostream& os) const;
    void save_(ostream& os, const TreeNode* node) const;

    friend class BasePredictor;
    static unique_ptr<BasePredictor> load_(istream& is, int version);
    static TreeNode* load_(istream& is, TreeNode* node, int version);

private:
    const TreeNode* const root_;
    const std::any nodeContainer_;
};

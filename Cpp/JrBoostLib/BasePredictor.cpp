//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "BasePredictor.h"

#include "Base128Encoding.h"
#include "OmpParallel.h"
#include "Tree.h"


void BasePredictor::predict(CRefXXfc inData, double c, RefXd outData) const
{
    PROFILE::PUSH(PROFILE::ZERO);   // calibrate the profiling
    PROFILE::SWITCH(PROFILE::PREDICT);

    if (::currentInterruptHandler != nullptr)
        ::currentInterruptHandler->check();
    if (abortThreads)
        throw ThreadAborted();

    predict_(inData, c, outData);

    PROFILE::POP();
}


unique_ptr<BasePredictor> BasePredictor::load_(istream& is, int version)
{
    int type = is.get();
    if (version < 6) {
        if (type == 100)
            return ConstantPredictor::load_(is, version);
        if (type == 101)
            return StumpPredictor::load_(is, version);
        if (version >= 2 && type == 102)
            return TreePredictor::load_(is, version);
        if (version >= 4 && type == 103)
            return ForestPredictor::load_(is, version);
    }
    else {
        if (type == 'Z')
            return ZeroPredictor::load_(is, version);
        if (type == 'C')
            return ConstantPredictor::load_(is, version);
        if (type == 'S')
            return StumpPredictor::load_(is, version);
        if (type == 'T')
            return TreePredictor::load_(is, version);
        if (type == 'F')
            return ForestPredictor::load_(is, version);
    }
    parseError(is);
}

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<BasePredictor> ZeroPredictor::createInstance() { return makeUnique<ZeroPredictor>(); }

void ZeroPredictor::predict_(CRefXXfc /*inData*/, double /*c*/, RefXd /*outData*/) const {}

double ZeroPredictor::predictOne_(CRefXf inData) const { return 0.0; }

size_t ZeroPredictor::variableCount_() const { return 0; }

void ZeroPredictor::variableWeights_(double /*c*/, RefXd /*weights*/) const {}

unique_ptr<BasePredictor> ZeroPredictor::reindexVariables_(CRefXs /*newIndices*/) const { return createInstance(); }

void ZeroPredictor::save_(ostream& os) const { os.put('Z'); }

unique_ptr<BasePredictor> ZeroPredictor::load_(istream& /*is*/, int /*version*/) { return createInstance(); }

//----------------------------------------------------------------------------------------------------------------------

ConstantPredictor::ConstantPredictor(double y) : y_(static_cast<float>(y)) { ASSERT(std::isfinite(y)); }

unique_ptr<BasePredictor> ConstantPredictor::createInstance(double y)
{
    if (y == 0.0)
        return ZeroPredictor::createInstance();
    return makeUnique<ConstantPredictor>(y);
}

void ConstantPredictor::predict_(CRefXXfc /*inData*/, double c, RefXd outData) const { outData += c * y_; }

double ConstantPredictor::predictOne_(CRefXf inData) const { return y_; }

size_t ConstantPredictor::variableCount_() const { return 0; }

void ConstantPredictor::variableWeights_(double /*c*/, RefXd /*weights*/) const {}

unique_ptr<BasePredictor> ConstantPredictor::reindexVariables_(CRefXs /*newIndices*/) const
{
    return createInstance(y_);
}

void ConstantPredictor::save_(ostream& os) const
{
    os.put('C');
    os.write(reinterpret_cast<const char*>(&y_), sizeof(y_));
}

unique_ptr<BasePredictor> ConstantPredictor::load_(istream& is, int version)
{
    if (version < 2)
        is.get();
    float y;
    is.read(reinterpret_cast<char*>(&y), sizeof(y));
    return createInstance(y);
}

//----------------------------------------------------------------------------------------------------------------------

StumpPredictor::StumpPredictor(size_t j, float x, float leftY, float rightY, float gain) :
    j_{j}, x_{x}, leftY_{leftY}, rightY_{rightY}, gain_{gain}
{
    ASSERT(std::isfinite(x) && std::isfinite(leftY) && std::isfinite(rightY));
}

unique_ptr<BasePredictor> StumpPredictor::createInstance(size_t j, float x, float leftY, float rightY, float gain)
{
    return makeUnique<StumpPredictor>(j, x, leftY, rightY, gain);
}

void StumpPredictor::predict_(CRefXXfc inData, double c, RefXd outData) const
{
    const size_t sampleCount = inData.rows();
    for (size_t i = 0; i != sampleCount; ++i) {
        double y = (inData(i, j_) < x_) ? leftY_ : rightY_;
        outData(i) += c * y;
    }
}

double StumpPredictor::predictOne_(CRefXf inData) const { return (inData(j_) < x_) ? leftY_ : rightY_; }

size_t StumpPredictor::variableCount_() const { return j_ + 1; }

void StumpPredictor::variableWeights_(double c, RefXd weights) const { weights(j_) += c * gain_; }

unique_ptr<BasePredictor> StumpPredictor::reindexVariables_(CRefXs newIndices) const
{
    return createInstance(newIndices(j_), x_, leftY_, rightY_, gain_);
}

void StumpPredictor::save_(ostream& os) const
{
    os.put('S');
    base128Save(os, j_);
    os.write(reinterpret_cast<const char*>(&x_), sizeof(x_));
    os.write(reinterpret_cast<const char*>(&leftY_), sizeof(leftY_));
    os.write(reinterpret_cast<const char*>(&rightY_), sizeof(rightY_));
    os.write(reinterpret_cast<const char*>(&gain_), sizeof(gain_));
}

unique_ptr<BasePredictor> StumpPredictor::load_(istream& is, int version)
{
    if (version < 2)
        is.get();

    size_t j;
    float x;
    float leftY;
    float rightY;
    float gain;

    if (version >= 5)
        j = base128Load(is);
    else {
        uint32_t j32;
        is.read(reinterpret_cast<char*>(&j32), sizeof(j32));
        j = static_cast<uint64_t>(j32);
    }

    is.read(reinterpret_cast<char*>(&x), sizeof(x));
    is.read(reinterpret_cast<char*>(&leftY), sizeof(leftY));
    is.read(reinterpret_cast<char*>(&rightY), sizeof(rightY));

    if (version >= 3 && version < 5)
        is.read(reinterpret_cast<char*>(&gain), sizeof(gain));
    else if (version < 8)
        gain = numeric_limits<float>::quiet_NaN();
    else
        is.read(reinterpret_cast<char*>(&gain), sizeof(gain));

    return createInstance(j, x, leftY, rightY, gain);
}

//----------------------------------------------------------------------------------------------------------------------

TreePredictor::TreePredictor(const TreeNode* root) : nodes_(TreeTools::cloneTree(root)) {}

TreePredictor::TreePredictor(vector<TreeNode>&& nodes) : nodes_(move(nodes)) {}

unique_ptr<BasePredictor> TreePredictor::createInstance(const TreeNode* root)
{
    const size_t treeDepth = TreeTools::treeDepth(root);

    if (treeDepth == 0)
        return ConstantPredictor::createInstance(root->y);

    if (treeDepth == 1)
        return StumpPredictor::createInstance(root->j, root->x, root->leftChild->y, root->rightChild->y, root->gain);

    return makeUnique<TreePredictor>(root);
}

unique_ptr<BasePredictor> TreePredictor::createInstance(vector<TreeNode>&& nodes)
{
    return makeUnique<TreePredictor>(move(nodes));
}

void TreePredictor::predict_(CRefXXfc inData, double c, RefXd outData) const
{
    const TreeNode* root = data(nodes_);
    TreeTools::predict(root, inData, c, outData);
}

double TreePredictor::predictOne_(CRefXf inData) const
{
    const TreeNode* root = data(nodes_);
    return TreeTools::predictOne(root, inData);
}

size_t TreePredictor::variableCount_() const
{
    const TreeNode* root = data(nodes_);
    return TreeTools::variableCount(root);
}

void TreePredictor::variableWeights_(double c, RefXd weights) const
{
    const TreeNode* root = data(nodes_);
    TreeTools::variableWeights(root, c, weights);
}

unique_ptr<BasePredictor> TreePredictor::reindexVariables_(CRefXs newIndices) const
{
    const TreeNode* root = data(nodes_);
    vector<TreeNode> nodes = TreeTools::reindexTree(root, newIndices);
    return createInstance(move(nodes));
}

void TreePredictor::save_(ostream& os) const
{
    os.put('T');
    const TreeNode* root = data(nodes_);
    TreeTools::saveTree(root, os);
}

unique_ptr<BasePredictor> TreePredictor::load_(istream& is, int version)
{
    vector<TreeNode> nodes = TreeTools::loadTree(is, version);
    return makeUnique<TreePredictor>(move(nodes));
}

//----------------------------------------------------------------------------------------------------------------------

ForestPredictor::ForestPredictor(vector<unique_ptr<BasePredictor>>&& basePredictors) :
    basePredictors_(move(basePredictors))
{
}

unique_ptr<BasePredictor> ForestPredictor::createInstance(vector<unique_ptr<BasePredictor>>&& basePredictors)
{
    return makeUnique<ForestPredictor>(move(basePredictors));
}

void ForestPredictor::predict_(CRefXXfc inData, double c, RefXd outData) const
{
    c /= size(basePredictors_);
    for (const auto& basePredictor : basePredictors_)
        basePredictor->predict_(inData, c, outData);
}

double ForestPredictor::predictOne_(CRefXf inData) const
{
    double pred = 0;
    for (const auto& basePredictor : basePredictors_)
        pred += basePredictor->predictOne_(inData);
    pred /= size(basePredictors_);
    return pred;
}

size_t ForestPredictor::variableCount_() const
{
    size_t n = 0;
    for (const auto& basePredictor : basePredictors_)
        n = std::max(n, basePredictor->variableCount_());
    return n;
}

void ForestPredictor::variableWeights_(double c, RefXd weights) const
{
    c /= size(basePredictors_);
    for (const auto& basePredictor : basePredictors_)
        basePredictor->variableWeights_(c, weights);
}

unique_ptr<BasePredictor> ForestPredictor::reindexVariables_(CRefXs newIndices) const
{
    vector<unique_ptr<BasePredictor>> basePredictors;
    basePredictors.reserve(size(basePredictors_));
    for (const auto& basePredictor : basePredictors_)
        basePredictors.push_back(basePredictor->reindexVariables_(newIndices));
    return createInstance(move(basePredictors));
}

void ForestPredictor::save_(ostream& os) const
{
    os.put('F');
    base128Save(os, size(basePredictors_));
    for (const auto& basePredictor : basePredictors_)
        basePredictor->save_(os);
}

unique_ptr<BasePredictor> ForestPredictor::load_(istream& is, int version)
{
    size_t n;
    if (version < 5) {
        uint32_t n32;
        is.read(reinterpret_cast<char*>(&n32), sizeof(n32));
        n = static_cast<uint64_t>(n32);
    }
    else
        n = base128Load(is);

    vector<unique_ptr<BasePredictor>> basePredictors;
    basePredictors.reserve(n);
    for (; n != 0; --n)
        basePredictors.push_back(BasePredictor::load_(is, version));

    return createInstance(move(basePredictors));
}

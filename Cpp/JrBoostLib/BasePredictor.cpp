//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "BasePredictor.h"

#include "Base128Encoding.h"
#include "Tree.h"


unique_ptr<BasePredictor> BasePredictor::load(istream& is, int version)
{
    int type = is.get();
    if (version < 6) {
        if (type == 100)
            return ConstantPredictor::load(is, version);
        if (type == 101)
            return StumpPredictor::load(is, version);
        if (version >= 2 && type == 102)
            return TreePredictor::load(is, version);
        if (version >= 4 && type == 103)
            return ForestPredictor::load(is, version);
    }
    else {
        if (type == 'Z')
            return ZeroPredictor::load(is, version);
        if (type == 'C')
            return ConstantPredictor::load(is, version);
        if (type == 'S')
            return StumpPredictor::load(is, version);
        if (type == 'T')
            return TreePredictor::load(is, version);
        if (type == 'F')
            return ForestPredictor::load(is, version);
    }
    parseError(is);
}

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<ZeroPredictor> ZeroPredictor::createInstance()
{
    return makeUnique<ZeroPredictor>();
}

void ZeroPredictor::predict(CRefXXfc /*inData*/, double /*c*/, RefXd /*outData*/) const
{}

void ZeroPredictor::variableWeights(double /*c*/, RefXd /*weights*/) const
{}

unique_ptr<BasePredictor> ZeroPredictor::reindexVariables(const vector<size_t>& /*newIndices*/) const
{
    return createInstance();
}

void ZeroPredictor::save(ostream& os) const
{
    os.put('Z');
}

unique_ptr<ZeroPredictor> ZeroPredictor::load(istream& /*is*/, int /*version*/)
{
    return createInstance();
}

//----------------------------------------------------------------------------------------------------------------------

ConstantPredictor::ConstantPredictor(double y) :
    y_(static_cast<float>(y))
{
    ASSERT(std::isfinite(y));
}

unique_ptr<ConstantPredictor> ConstantPredictor::createInstance(double y)
{
    return makeUnique<ConstantPredictor>(y);
}

void ConstantPredictor::predict(CRefXXfc /*inData*/, double c, RefXd outData) const
{
    outData += c * static_cast<double>(y_);
}

void ConstantPredictor::variableWeights(double /*c*/, RefXd /*weights*/) const
{
}

unique_ptr<BasePredictor> ConstantPredictor::reindexVariables(const vector<size_t>& /*newIndices*/) const
{
    return createInstance(y_);
}

void ConstantPredictor::save(ostream& os) const
{
    os.put('C');
    os.write(reinterpret_cast<const char*>(&y_), sizeof(y_));
}

unique_ptr<ConstantPredictor> ConstantPredictor::load(istream& is, int version)
{
    if (version < 2) is.get();
    float y;
    is.read(reinterpret_cast<char*>(&y), sizeof(y));
    return createInstance(y);
}

//----------------------------------------------------------------------------------------------------------------------

StumpPredictor::StumpPredictor(size_t j, float x, float leftY, float rightY, float gain) :
    j_{ j },
    x_{ x },
    leftY_{ leftY },
    rightY_{ rightY },
    gain_{ gain }
{
    ASSERT(std::isfinite(x) && std::isfinite(leftY) && std::isfinite(rightY));
}

unique_ptr<StumpPredictor> StumpPredictor::createInstance(size_t j, float x, float leftY, float rightY, float gain)
{
    return makeUnique<StumpPredictor>(j, x, leftY, rightY, gain);
}

void StumpPredictor::predict(CRefXXfc inData, double c, RefXd outData) const
{
    const size_t sampleCount = inData.rows();
    for (size_t i = 0; i != sampleCount; ++i) {
        double y = (inData(i, j_) < x_) ? leftY_ : rightY_;
        outData(i) += c * y;
    }
}

void StumpPredictor::variableWeights(double c, RefXd weights) const
{
    weights(j_) += c * gain_;
}

unique_ptr<BasePredictor> StumpPredictor::reindexVariables(const vector<size_t>& newIndices) const
{
    return createInstance(newIndices[j_], x_, leftY_, rightY_, gain_);
}

void StumpPredictor::save(ostream& os) const
{
    os.put('S');
    base128Save(os, j_);
    os.write(reinterpret_cast<const char*>(&x_), sizeof(x_));
    os.write(reinterpret_cast<const char*>(&leftY_), sizeof(leftY_));
    os.write(reinterpret_cast<const char*>(&rightY_), sizeof(rightY_));
}

unique_ptr<StumpPredictor> StumpPredictor::load(istream& is, int version)
{
    if (version < 2) is.get();

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
    else
        gain = numeric_limits<float>::quiet_NaN();

    return createInstance(j, x, leftY, rightY, gain);
}

//----------------------------------------------------------------------------------------------------------------------

TreePredictor::TreePredictor(const TreeNode* root) :
    nodes_(TreeTools::cloneTree(root))
{}

TreePredictor::TreePredictor(vector<TreeNode>&& nodes) :
    nodes_(move(nodes))
{}

unique_ptr<TreePredictor> TreePredictor::createInstance(const TreeNode* root)
{
    return makeUnique<TreePredictor>(root);
}

unique_ptr<TreePredictor> TreePredictor::createInstance(vector<TreeNode>&& nodes)
{
    return makeUnique<TreePredictor>(move(nodes));
}

void TreePredictor::predict(CRefXXfc inData, double c, RefXd outData) const
{
    const TreeNode* root = data(nodes_);
    TreeTools::predict(root, inData, c, outData);
}

void TreePredictor::variableWeights(double c, RefXd weights) const
{
    const TreeNode* root = data(nodes_);
    TreeTools::variableWeights(root, c, weights);
}

unique_ptr<BasePredictor> TreePredictor::reindexVariables(const vector<size_t>& newIndices) const
{
    const TreeNode* root = data(nodes_);
    vector<TreeNode> nodes = TreeTools::reindexTree(root, newIndices);
    return createInstance(move(nodes));
}

void TreePredictor::save(ostream& os) const
{
    os.put('T');
    const TreeNode* root = data(nodes_);
    TreeTools::saveTree(root, os);
}

unique_ptr<TreePredictor> TreePredictor::load(istream& is, int version)
{
    vector<TreeNode> nodes = TreeTools::loadTree(is, version);
    return makeUnique<TreePredictor>(move(nodes));
}

//----------------------------------------------------------------------------------------------------------------------

ForestPredictor::ForestPredictor(vector<unique_ptr<BasePredictor>>&& basePredictors) :
    basePredictors_(move(basePredictors))
{}

unique_ptr<ForestPredictor> ForestPredictor::createInstance(vector<unique_ptr<BasePredictor>>&& basePredictors)
{
    return makeUnique<ForestPredictor>(move(basePredictors));
}

void ForestPredictor::predict(CRefXXfc inData, double c, RefXd outData) const
{
    c /= size(basePredictors_);
    for (const auto& basePredictor : basePredictors_)
        basePredictor->predict(inData, c, outData);
}

void ForestPredictor::variableWeights(double c, RefXd weights) const
{
    c /= size(basePredictors_);
    for (const auto& basePredictor : basePredictors_)
        basePredictor->variableWeights(c, weights);
}

unique_ptr<BasePredictor> ForestPredictor::reindexVariables(const vector<size_t>& newIndices) const
{
    vector<unique_ptr<BasePredictor>> basePredictors;
    basePredictors.reserve(size(basePredictors_));
    for (const auto& basePredictor: basePredictors_)
        basePredictors.push_back(basePredictor->reindexVariables(newIndices));
    return createInstance(move(basePredictors));
}

void ForestPredictor::save(ostream& os) const
{
    os.put('F');
    base128Save(os, size(basePredictors_));
    for (const auto& basePredictor : basePredictors_)
        basePredictor->save(os);
}

unique_ptr<ForestPredictor> ForestPredictor::load(istream& is, int version)
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
        basePredictors.push_back(BasePredictor::load(is, version));

    return createInstance(move(basePredictors));
}

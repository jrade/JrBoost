//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

struct TreeNode;


class BasePredictor
{
public:
    virtual ~BasePredictor() = default;

    // make a prediction based on inData
    // add the prediction, multiplied by c, to outData
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const = 0;

    // add the variable importance weights, multiplied by c, to weights
    virtual void variableWeights(double c, RefXd weights) const = 0;

    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const = 0;

    virtual void save(ostream& os) const = 0;

    static unique_ptr<BasePredictor> load(istream& is, int version);

protected:
    BasePredictor() = default;
    
// deleted:
    BasePredictor(const BasePredictor&) = delete;
    BasePredictor& operator=(const BasePredictor&) = delete;
};

//----------------------------------------------------------------------------------------------------------------------

class ZeroPredictor : public BasePredictor
{
public:
    virtual ~ZeroPredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<ZeroPredictor> createInstance();
    static unique_ptr<ZeroPredictor> load(istream& is, int version);

private:
    ZeroPredictor() = default;

    friend class MakeUniqueHelper<ZeroPredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class ConstantPredictor : public BasePredictor
{
public:
    virtual ~ConstantPredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<ConstantPredictor> createInstance(double y);
    static unique_ptr<ConstantPredictor> load(istream& is, int version);

private:
    ConstantPredictor(double y);

    float y_;

    friend class MakeUniqueHelper<ConstantPredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class StumpPredictor : public BasePredictor
{
public:
    virtual ~StumpPredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<StumpPredictor> createInstance(size_t j, float x, float leftY, float rightY, float gain);
    static unique_ptr<StumpPredictor> load(istream& is, int version);

private:
    StumpPredictor(size_t j, float x, float leftY, float rightY, float gain);

    size_t j_;
    float x_;
    float leftY_;
    float rightY_;
    float gain_;

    friend class MakeUniqueHelper<StumpPredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class TreePredictor : public BasePredictor
{
public:
    virtual ~TreePredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<TreePredictor> createInstance(const TreeNode* root);
    static unique_ptr<TreePredictor> createInstance(vector<TreeNode>&& nodes);
    static unique_ptr<TreePredictor> load(istream& is, int version);

private:
    TreePredictor(const TreeNode* root);
    TreePredictor(vector<TreeNode>&& nodes);

    const vector<TreeNode> nodes_;

    friend class MakeUniqueHelper<TreePredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class ForestPredictor : public BasePredictor
{
public:
    virtual ~ForestPredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<ForestPredictor> createInstance(vector<unique_ptr<BasePredictor>>&& basePredictors);
    static unique_ptr<ForestPredictor> load(istream& is, int version);

private:
    ForestPredictor(vector<unique_ptr<BasePredictor>>&& basePredictors);

    const vector<unique_ptr<BasePredictor>> basePredictors_;

    friend class MakeUniqueHelper<ForestPredictor>;
};

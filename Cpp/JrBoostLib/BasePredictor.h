//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

struct TreeNode;


class BasePredictor {   // abstract class
public:
    virtual ~BasePredictor() = default;

    // make a prediction based on inData
    // add the prediction, multiplied by c, to outData
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const = 0;
    virtual double predictOne(CRefXf inData) const = 0;

    virtual size_t variableCount() const = 0;
    // add the variable importance weights, multiplied by c, to weights
    virtual void variableWeights(double c, RefXd weights) const = 0;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const = 0;

    virtual void save(ostream& os) const = 0;

    static unique_ptr<BasePredictor> load(istream& is, int version);

protected:
    BasePredictor() = default;
    BasePredictor(const BasePredictor&) = delete;
    BasePredictor& operator=(const BasePredictor&) = delete;
};

//----------------------------------------------------------------------------------------------------------------------

class ZeroPredictor : public BasePredictor {   // immutable class
public:
    virtual ~ZeroPredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne(CRefXf inData) const;
    virtual size_t variableCount() const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<BasePredictor> createInstance();
    static unique_ptr<BasePredictor> load(istream& is, int version);

private:
    ZeroPredictor() = default;

    friend class MakeUniqueHelper<ZeroPredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class ConstantPredictor : public BasePredictor {   // immutable class
public:
    virtual ~ConstantPredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne(CRefXf inData) const;
    virtual size_t variableCount() const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<BasePredictor> createInstance(double y);
    static unique_ptr<BasePredictor> load(istream& is, int version);

private:
    ConstantPredictor(double y);

    float y_;

    friend class MakeUniqueHelper<ConstantPredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class StumpPredictor : public BasePredictor {   // immutable class
public:
    virtual ~StumpPredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne(CRefXf inData) const;
    virtual size_t variableCount() const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<BasePredictor> createInstance(size_t j, float x, float leftY, float rightY, float gain);
    static unique_ptr<BasePredictor> load(istream& is, int version);

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

class TreePredictor : public BasePredictor {   // immutable class
public:
    virtual ~TreePredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne(CRefXf inData) const;
    virtual size_t variableCount() const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<BasePredictor> createInstance(const TreeNode* root);
    static unique_ptr<BasePredictor> createInstance(vector<TreeNode>&& nodes);
    static unique_ptr<BasePredictor> load(istream& is, int version);

private:
    TreePredictor(const TreeNode* root);
    TreePredictor(vector<TreeNode>&& nodes);

    const vector<TreeNode> nodes_;

    friend class MakeUniqueHelper<TreePredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class ForestPredictor : public BasePredictor {
public:
    virtual ~ForestPredictor() = default;
    virtual void predict(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne(CRefXf inData) const;
    virtual size_t variableCount() const;
    virtual void variableWeights(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables(const vector<size_t>& newIndices) const;
    virtual void save(ostream& os) const;

    static unique_ptr<BasePredictor> createInstance(vector<unique_ptr<BasePredictor>>&& basePredictors);
    static unique_ptr<BasePredictor> load(istream& is, int version);

private:
    ForestPredictor(vector<unique_ptr<BasePredictor>>&& basePredictors);

    const vector<unique_ptr<BasePredictor>> basePredictors_;

    friend class MakeUniqueHelper<ForestPredictor>;
};

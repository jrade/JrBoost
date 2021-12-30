//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

struct TreeNode;


class BasePredictor {   // abstract class
public:
    virtual ~BasePredictor() = default;

    // make a prediction based on inData
    // add the prediction, multiplied by c, to outData
    void predict(CRefXXfc inData, double c, RefXd outData) const;

protected:
    BasePredictor() = default;
    BasePredictor(const BasePredictor&) = delete;
    BasePredictor& operator=(const BasePredictor&) = delete;

private:
    virtual void predict_(CRefXXfc inData, double c, RefXd outData) const = 0;
    virtual double predictOne_(CRefXf inData) const = 0;
    virtual size_t variableCount_() const = 0;
    // add the variable importance weights, multiplied by c, to weights
    virtual void variableWeights_(double c, RefXd weights) const = 0;
    virtual unique_ptr<BasePredictor> reindexVariables_(const vector<size_t>& newIndices) const = 0;
    virtual void save_(ostream& os) const = 0;

    static unique_ptr<BasePredictor> load_(istream& is, int version);

private:
    friend class ForestPredictor;
    friend class BoostPredictor;
};

//----------------------------------------------------------------------------------------------------------------------

class ZeroPredictor : public BasePredictor {   // immutable class
public:
    static unique_ptr<BasePredictor> createInstance();
    virtual ~ZeroPredictor() = default;

private:
    ZeroPredictor() = default;

    virtual void predict_(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne_(CRefXf inData) const;
    virtual size_t variableCount_() const;
    virtual void variableWeights_(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables_(const vector<size_t>& newIndices) const;
    virtual void save_(ostream& os) const;

    static unique_ptr<BasePredictor> load_(istream& is, int version);

private:
    friend class MakeUniqueHelper<ZeroPredictor>;
    friend class BasePredictor;
};

//----------------------------------------------------------------------------------------------------------------------

class ConstantPredictor : public BasePredictor {   // immutable class
public:
    static unique_ptr<BasePredictor> createInstance(double y);
    virtual ~ConstantPredictor() = default;

private:
    ConstantPredictor(double y);

    virtual void predict_(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne_(CRefXf inData) const;
    virtual size_t variableCount_() const;
    virtual void variableWeights_(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables_(const vector<size_t>& newIndices) const;
    virtual void save_(ostream& os) const;

    static unique_ptr<BasePredictor> load_(istream& is, int version);

private:
    float y_;

private:
    friend class MakeUniqueHelper<ConstantPredictor>;
    friend class BasePredictor;
};

//----------------------------------------------------------------------------------------------------------------------

class StumpPredictor : public BasePredictor {   // immutable class
public:
    static unique_ptr<BasePredictor> createInstance(size_t j, float x, float leftY, float rightY, float gain);
    virtual ~StumpPredictor() = default;

private:
    StumpPredictor(size_t j, float x, float leftY, float rightY, float gain);

    virtual void predict_(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne_(CRefXf inData) const;
    virtual size_t variableCount_() const;
    virtual void variableWeights_(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables_(const vector<size_t>& newIndices) const;
    virtual void save_(ostream& os) const;

    static unique_ptr<BasePredictor> load_(istream& is, int version);

private:
    size_t j_;
    float x_;
    float leftY_;
    float rightY_;
    float gain_;

private:
    friend class MakeUniqueHelper<StumpPredictor>;
    friend class BasePredictor;
};

//----------------------------------------------------------------------------------------------------------------------

class TreePredictor : public BasePredictor {   // immutable class
public:
    static unique_ptr<BasePredictor> createInstance(const TreeNode* root);
    static unique_ptr<BasePredictor> createInstance(vector<TreeNode>&& nodes);
    virtual ~TreePredictor() = default;

private:
    TreePredictor(const TreeNode* root);
    TreePredictor(vector<TreeNode>&& nodes);

    virtual void predict_(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne_(CRefXf inData) const;
    virtual size_t variableCount_() const;
    virtual void variableWeights_(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables_(const vector<size_t>& newIndices) const;
    virtual void save_(ostream& os) const;

    static unique_ptr<BasePredictor> load_(istream& is, int version);

private:
    const vector<TreeNode> nodes_;

private:
    friend class MakeUniqueHelper<TreePredictor>;
    friend class BasePredictor;
};

//----------------------------------------------------------------------------------------------------------------------

class ForestPredictor : public BasePredictor {
public:
    static unique_ptr<BasePredictor> createInstance(vector<unique_ptr<BasePredictor>>&& basePredictors);
    virtual ~ForestPredictor() = default;

private:
    ForestPredictor(vector<unique_ptr<BasePredictor>>&& basePredictors);

    virtual void predict_(CRefXXfc inData, double c, RefXd outData) const;
    virtual double predictOne_(CRefXf inData) const;
    virtual size_t variableCount_() const;
    virtual void variableWeights_(double c, RefXd weights) const;
    virtual unique_ptr<BasePredictor> reindexVariables_(const vector<size_t>& newIndices) const;
    virtual void save_(ostream& os) const;

    static unique_ptr<BasePredictor> load_(istream& is, int version);

private:
    const vector<unique_ptr<BasePredictor>> basePredictors_;

private:
    friend class MakeUniqueHelper<ForestPredictor>;
    friend class BasePredictor;
};

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class BasePredictor;

// File format versions:
// 1 - original version
// 2 - added tree predictors, simplified version handling
// 3 - added gain to stump and tree predictors
// 4 - added forest predictors
// 5 - Base128 encoded integers, removed gain from stump and tree predictors
// 6 - changed predictor and base predictor tags
// 7 - added union predictors

class Predictor
{
public:
    virtual ~Predictor() = default;
    size_t variableCount() const { return variableCount_; }
    ArrayXd predict(CRefXXfc inData) const;
    ArrayXd variableWeights() const;
    shared_ptr<Predictor> reindexVariables(const vector<size_t>& newIndices) const;
    void save(const string& filePath) const;
    void save(ostream& os) const;
    static shared_ptr<Predictor> load(const string& filePath);
    static shared_ptr<Predictor> load(istream& is);

protected:
    Predictor(size_t variableCount) : variableCount_(variableCount) {}
    static size_t maxVariableCount_(const vector<shared_ptr<Predictor>>& predictors);
    
// deleted:
    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor&) = delete;

private:
    virtual ArrayXd predictImpl_(CRefXXfc inData) const = 0;
    // add the variable importance weights, multiplied by c, to weights
    virtual void variableWeightsImpl_(double c, RefXd weights) const = 0;
    virtual shared_ptr<Predictor> reindexVariablesImpl_(const vector<size_t>& newIndices, size_t variableCount) const = 0;
    virtual void saveBody_(ostream& os) const = 0;
    static int loadHeader_(istream& is);
    static shared_ptr<Predictor> loadBody_(istream& is, int version);

    const size_t variableCount_;
    static const int currentFileFormatVersion_ = 7;

    friend class EnsemblePredictor;
    friend class UnionPredictor;
};

//----------------------------------------------------------------------------------------------------------------------

class BoostPredictor : public Predictor
{
public:
    virtual ~BoostPredictor();
    static shared_ptr<BoostPredictor> createInstance(
        size_t variableCount,
        double c0,
        double c1,
        vector<unique_ptr<BasePredictor>>&& basePredictors
    );

private:
    BoostPredictor(
        size_t variableCount,
        double c0,
        double c1,
        vector<unique_ptr<BasePredictor>>&& basePredictors
    );
    virtual ArrayXd predictImpl_(CRefXXfc inData) const;
    virtual void variableWeightsImpl_(double c, RefXd weights) const;
    virtual shared_ptr<Predictor> reindexVariablesImpl_(const vector<size_t>& newIndices, size_t variableCount) const;
    virtual void saveBody_(ostream& os) const;
    static shared_ptr<BoostPredictor> loadBody_(istream& is, int version);

    float c0_;
    float c1_;
    vector<unique_ptr<BasePredictor>> basePredictors_;

    friend class Predictor;
    friend class MakeSharedHelper<BoostPredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class EnsemblePredictor : public Predictor
{
public:
    virtual ~EnsemblePredictor() = default;
    static shared_ptr<EnsemblePredictor> createInstance(const vector<shared_ptr<Predictor>>& predictors);

private:
    EnsemblePredictor(const vector<shared_ptr<Predictor>>& predictors);
    virtual ArrayXd predictImpl_(CRefXXfc inData) const;
    virtual void variableWeightsImpl_(double c, RefXd weights) const;
    virtual shared_ptr<Predictor> reindexVariablesImpl_(const vector<size_t>& newIndices, size_t variableCount) const;
    virtual void saveBody_(ostream& os) const;
    static shared_ptr<EnsemblePredictor> loadBody_(istream& is, int version);

    vector<shared_ptr<Predictor>> predictors_;

    friend class Predictor;
    friend class MakeSharedHelper<EnsemblePredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class UnionPredictor : public Predictor
{
public:
    virtual ~UnionPredictor() = default;
    static shared_ptr<UnionPredictor> createInstance(const vector<shared_ptr<Predictor>>& predictors);

private:
    UnionPredictor(const vector<shared_ptr<Predictor>>& predictors);
    virtual ArrayXd predictImpl_(CRefXXfc inData) const;
    virtual void variableWeightsImpl_(double c, RefXd weights) const;
    virtual shared_ptr<Predictor> reindexVariablesImpl_(const vector<size_t>& newIndices, size_t variableCount) const;
    virtual void saveBody_(ostream& os) const;
    static shared_ptr<UnionPredictor> loadBody_(istream& is, int version);

    vector<shared_ptr<Predictor>> predictors_;

    friend class Predictor;
    friend class MakeSharedHelper<UnionPredictor>;
};

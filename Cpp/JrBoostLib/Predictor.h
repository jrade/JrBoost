//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
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
// 8 - added variable weights, simplified handling of variable count, reintroduced gain

class Predictor {   // abstract class
public:
    virtual ~Predictor() = default;

    size_t variableCount() const { return variableCount_; }
    ArrayXd predict(CRefXXfc inData, size_t threadCount = 0) const;
    double predictOne(CRefXf inData) const;
    ArrayXf variableWeights() const;
    shared_ptr<Predictor> reindexVariables(CRefXs newIndices) const;

    void save(const string& filePath) const;
    void save(ostream& os) const;
    static shared_ptr<Predictor> load(const string& filePath);
    static shared_ptr<Predictor> load(istream& is);

protected:
    Predictor(size_t variableCount);
    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor&) = delete;

private:
    virtual ArrayXd predictImpl_(CRefXXfc inData, size_t threadCount) const = 0;
    virtual ArrayXd predictImplNoThreads_(CRefXXfc inData) const = 0;
    virtual double predictOneImpl_(CRefXf inData) const = 0;
    virtual ArrayXf variableWeightsImpl_() const = 0;
    virtual shared_ptr<Predictor> reindexVariablesImpl_(CRefXs newIndices) const = 0;
    virtual void saveImpl_(ostream& os) const = 0;
    static shared_ptr<Predictor> loadImpl_(istream& is, int version);

    const size_t variableCount_;

    static const int currentFileFormatVersion_ = 8;

    friend class EnsemblePredictor;
    friend class UnionPredictor;
};

//----------------------------------------------------------------------------------------------------------------------

class BoostPredictor : public Predictor {   // immutable class
public:
    static shared_ptr<Predictor>
    createInstance(double c0, double c1, vector<unique_ptr<BasePredictor>>&& basePredictors);

private:
    BoostPredictor(double c0, double c1, vector<unique_ptr<BasePredictor>>&& basePredictors);
    static size_t initVariableCount_(const vector<unique_ptr<BasePredictor>>& basePredictors);

    virtual ~BoostPredictor();
    virtual ArrayXd predictImpl_(CRefXXfc inData, size_t threadCount) const;
    virtual ArrayXd predictImplNoThreads_(CRefXXfc inData) const;
    virtual double predictOneImpl_(CRefXf inData) const;
    virtual ArrayXf variableWeightsImpl_() const;
    virtual shared_ptr<Predictor> reindexVariablesImpl_(CRefXs newIndices) const;
    virtual void saveImpl_(ostream& os) const;
    static shared_ptr<Predictor> loadImpl_(istream& is, int version);

    float c0_;
    float c1_;
    vector<unique_ptr<BasePredictor>> basePredictors_;

    friend class Predictor;
    friend class MakeSharedHelper<BoostPredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class EnsemblePredictor : public Predictor {   // immutable class
public:
    static shared_ptr<Predictor> createInstance(const vector<shared_ptr<Predictor>>& predictors);

private:
    EnsemblePredictor(const vector<shared_ptr<Predictor>>& predictors);
    static size_t initVariableCount_(const vector<shared_ptr<Predictor>>& predictors);

    virtual ~EnsemblePredictor() = default;
    virtual ArrayXd predictImpl_(CRefXXfc inData, size_t threadCount) const;
    virtual ArrayXd predictImplNoThreads_(CRefXXfc inData) const;
    virtual double predictOneImpl_(CRefXf inData) const;
    virtual ArrayXf variableWeightsImpl_() const;
    virtual shared_ptr<Predictor> reindexVariablesImpl_(CRefXs newIndices) const;
    virtual void saveImpl_(ostream& os) const;
    static shared_ptr<Predictor> loadImpl_(istream& is, int version);

    vector<shared_ptr<Predictor>> predictors_;

    friend class Predictor;
    friend class MakeSharedHelper<EnsemblePredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class UnionPredictor : public Predictor {   // immutable class
public:
    static shared_ptr<Predictor> createInstance(const vector<shared_ptr<Predictor>>& predictors);

private:
    UnionPredictor(const vector<shared_ptr<Predictor>>& predictors);
    static size_t initVariableCount_(const vector<shared_ptr<Predictor>>& predictors);

    virtual ~UnionPredictor() = default;
    virtual ArrayXd predictImpl_(CRefXXfc inData, size_t threadCount) const;
    virtual ArrayXd predictImplNoThreads_(CRefXXfc inData) const;
    virtual double predictOneImpl_(CRefXf inData) const;
    virtual ArrayXf variableWeightsImpl_() const;
    virtual shared_ptr<Predictor> reindexVariablesImpl_(CRefXs newIndices) const;
    virtual void saveImpl_(ostream& os) const;
    static shared_ptr<Predictor> loadImpl_(istream& is, int version);

    vector<shared_ptr<Predictor>> predictors_;

    friend class Predictor;
    friend class MakeSharedHelper<UnionPredictor>;
};

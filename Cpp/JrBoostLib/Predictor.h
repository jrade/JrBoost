//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

class BasePredictor;


class Predictor
{
public:
    virtual ~Predictor() = default;
    size_t variableCount() const { return variableCount_; }
    ArrayXd predict(CRefXXfc inData) const;
    ArrayXd variableWeights() const;

    void save(const string& filePath) const;
    static shared_ptr<Predictor> load(const string& filePath);

    void save(ostream& os) const;
    static shared_ptr<Predictor> load(istream& is);

    // low level functions, but used by EnsemblePredictor
    virtual ArrayXd predictImpl(CRefXXfc inData) const = 0;
    // add the variable importance weights, multiplied by c, to weights
    virtual void variableWeightsImpl(double c, RefXd weights) const = 0;
    virtual void saveBody(ostream& os) const = 0;
    static shared_ptr<Predictor> loadBody(istream& is, int version);

protected:
    Predictor(size_t variableCount) : variableCount_(variableCount) {}

// deleted:
    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor&) = delete;

private:
    const size_t variableCount_;

    // File format versions:
    // 1 - original version
    // 2 - added tree predictors, simplified version handling
    // 3 - added gain to stump and tree predictors
    // 4 - added forest predictors
    // 5 - Base128 encoded integers, removed gain
    // 6 - changed predictor and base predictor tags
    static const int currentVersion_ = 6;

    static int loadHeader_(istream& is);
};

//----------------------------------------------------------------------------------------------------------------------

class BoostPredictor : public Predictor
{
public:
    virtual ~BoostPredictor();
    virtual ArrayXd predictImpl(CRefXXfc inData) const;
    virtual void variableWeightsImpl(double c, RefXd weights) const;
    virtual void saveBody(ostream& os) const;

    static shared_ptr<BoostPredictor> createInstance(
        size_t variableCount,
        double c0,
        double c1,
        vector<unique_ptr<BasePredictor>>&& basePredictors
    );
    static shared_ptr<Predictor> loadBody(istream& is, int version);

private:
    BoostPredictor(
        size_t variableCount,
        double c0,
        double c1,
        vector<unique_ptr<BasePredictor>>&& basePredictors
    );

    float c0_;
    float c1_;
    vector<unique_ptr<BasePredictor>> basePredictors_;

    friend class MakeSharedHelper<BoostPredictor>;
};

//----------------------------------------------------------------------------------------------------------------------

class EnsemblePredictor : public Predictor
{
public:
    virtual ~EnsemblePredictor() = default;
    virtual ArrayXd predictImpl(CRefXXfc inData) const;
    virtual void variableWeightsImpl(double c, RefXd weights) const;
    virtual void saveBody(ostream& os) const;

    static shared_ptr<EnsemblePredictor> createInstance(const vector<shared_ptr<Predictor>>& predictors);
    static shared_ptr<Predictor> loadBody(istream& is, int version);

private:
    EnsemblePredictor(const vector<shared_ptr<Predictor>>& predictors);

    vector<shared_ptr<Predictor>> predictors_;

    friend class MakeSharedHelper<EnsemblePredictor>;
};

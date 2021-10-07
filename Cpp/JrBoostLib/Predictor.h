//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class Predictor
{
public:
    virtual ~Predictor() = default;
    virtual size_t variableCount() const = 0;

    ArrayXd predict(CRefXXfc inData) const;

    ArrayXd variableWeights() const;

    void save(const string& filePath) const;
    void save(ostream& os) const;
    static shared_ptr<Predictor> load(const string& filePath);
    static shared_ptr<Predictor> load(istream& is);

protected:
    enum { Boost = 0, Ensemble = 1 };

    Predictor() = default;

    static void parseError_ [[noreturn]] (istream& s);

private:
    friend class EnsemblePredictor;

    virtual ArrayXd predict_(CRefXXfc inData) const = 0;
    virtual void variableWeights_(double c, RefXd weights) const = 0;

    // File format versions:
    // 1 - original version
    // 2 - added tree predictors, simplified version handling
    // 3 - added gain to stump and tree predictors
    static const int currentVersion_ = 3;

    virtual void save_(ostream& os) const = 0;
    static shared_ptr<Predictor> load_(istream& is, int version);
    
// deleted:
    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor&) = delete;
};

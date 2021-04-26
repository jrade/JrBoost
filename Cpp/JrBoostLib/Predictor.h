//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class Predictor {
public:
    virtual ~Predictor() = default;
    virtual size_t variableCount() const = 0;

    ArrayXd predict(CRefXXf inData) const;

    void save(const string& filePath) const;
    void save(ostream& os) const;
    static shared_ptr<Predictor> load(const string& filePath);
    static shared_ptr<Predictor> load(istream& is);

protected:
    enum { Boost = 0, Ensemble = 1 };

    Predictor() = default;

private:
    virtual void save_(ostream& os) const = 0;

    friend class EnsemblePredictor;
    virtual ArrayXd predict_(CRefXXf inData) const = 0;
    static shared_ptr<Predictor> load_(istream& is);
    
// deleted:
    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor&) = delete;
};

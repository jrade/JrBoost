//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class Predictor {
public:
    virtual ~Predictor() = default;
    virtual size_t variableCount() const = 0;
    virtual void save(ostream& os) const = 0;

    void save(const string& filePath) const;
    ArrayXd predict(CRefXXf inData) const;

    static shared_ptr<Predictor> load(const string& filePath);
    static shared_ptr<Predictor> load(istream& is);

protected:
    enum { Boost = 0, Ensemble = 1 };

    Predictor() = default;


private:
    friend class EnsemblePredictor;
    virtual ArrayXd predictImpl_(CRefXXf inData) const = 0;

    void validateInData_(CRefXXf inData) const;
    
    // deleted:
    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor&) = delete;
};

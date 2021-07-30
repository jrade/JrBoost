//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class BasePredictor {
public:
    virtual ~BasePredictor() = default;

    void predict(CRefXXf inData, double c, RefXd outData) const;

protected:
    enum { Trivial = 100, Stump = 101, Tree = 102 };

    BasePredictor() = default;

    static void parseError_ [[noreturn]] (istream& is);

private:
    friend class BoostPredictor;
    virtual void predict_(CRefXXf inData, double c, RefXd outData) const = 0;
    virtual void variableWeights_(vector<double>& weights, double c) const = 0;
    virtual void save_(ostream& os) const = 0;
    static unique_ptr<BasePredictor> load_(istream& is, int version);

// deleted:
    BasePredictor(const BasePredictor&) = delete;
    BasePredictor& operator=(const BasePredictor&) = delete;
};

//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class BasePredictor
{
public:
    virtual ~BasePredictor() = default;

    void predict(CRefXXfc inData, double c, RefXd outData) const;

protected:
    enum { Trivial = 100, Stump = 101, Tree = 102, Forest = 103 };

    BasePredictor() = default;

    static void parseError_ [[noreturn]] (istream& is);

private:
    friend class BoostPredictor;
    friend class ForestPredictor;

    virtual void predict_(CRefXXfc inData, double c, RefXd outData) const = 0;
    virtual void variableWeights_(double c, RefXd weights) const = 0;
    virtual void save_(ostream& os) const = 0;
    static unique_ptr<BasePredictor> load_(istream& is, int version);

// deleted:
    BasePredictor(const BasePredictor&) = delete;
    BasePredictor& operator=(const BasePredictor&) = delete;
};

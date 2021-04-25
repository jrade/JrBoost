//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class BasePredictor {
public:
    virtual ~BasePredictor() = default;
    virtual void save(ostream& os) const = 0;

    void predict(CRefXXf inData, double c, RefXd outData) const;

    static unique_ptr<BasePredictor> load(istream& is);

protected:
    enum { Trivial = 100, Stump = 101 };

    BasePredictor() = default;

private:
    friend class BoostPredictor;
    virtual void predictImpl_(CRefXXf inData, double c, RefXd outData) const = 0;
    
// deleted:
    BasePredictor(const BasePredictor&) = delete;
    BasePredictor& operator=(const BasePredictor&) = delete;
};

#pragma once

class SimplePredictor;


class BoostPredictor {
public:
    virtual ~BoostPredictor();

    size_t variableCount() const { return variableCount_; }
    ArrayXd predict(CRefXXf inData) const;

// deleted:
    BoostPredictor(const BoostPredictor&) = delete;
    BoostPredictor& operator=(const BoostPredictor&) = delete;

private:
    friend class BoostTrainer;

    BoostPredictor(
        size_t variableCount,
        double c0,
        vector<double>&& c1,
        vector<unique_ptr<SimplePredictor>>&& basePredictors
    );

    void validateInData_(CRefXXf inData) const;

    size_t variableCount_;
    double c0_;
    vector<double> c1_;
    vector<unique_ptr<SimplePredictor>> basePredictors_;
};

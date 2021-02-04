#pragma once

class AbstractPredictor {
public:
    virtual ~AbstractPredictor() = default;
    size_t variableCount() const { return variableCount_; }
    virtual ArrayXd predict(CRefXXf inData) const = 0;
    virtual void predict(CRefXXf inData, double c, RefXd outData) const = 0;

protected:
    AbstractPredictor(size_t variableCount) : variableCount_(variableCount) {}

    void validateInData_(CRefXXf inData) const
    {
        const size_t variableCount = inData.cols();
        ASSERT(variableCount == variableCount_);
        //ASSERT((inData > -numeric_limits<float>::infinity()).all());
        //ASSERT((inData < numeric_limits<float>::infinity()).all());
    }

// deleted:
    AbstractPredictor(const AbstractPredictor&) = delete;
    AbstractPredictor& operator=(const AbstractPredictor&) = delete;

private:
    size_t variableCount_;
};

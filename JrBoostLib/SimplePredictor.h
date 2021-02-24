#pragma once

class SimplePredictor {
public:
    virtual ~SimplePredictor() = default;

    size_t variableCount() const { return variableCount_; }

    virtual void predict(CRefXXf inData, double c, RefXd outData) const = 0;

protected:
    SimplePredictor(size_t variableCount) : variableCount_(variableCount) {}

    void validateInData_(CRefXXf inData) const
    {
        const size_t variableCount = inData.cols();
        ASSERT(variableCount == variableCount_);
        ASSERT((inData > -numeric_limits<float>::infinity()).all());
        ASSERT((inData < numeric_limits<float>::infinity()).all());
    }

// deleted:
    SimplePredictor(const SimplePredictor&) = delete;
    SimplePredictor& operator=(const SimplePredictor&) = delete;

private:
    size_t variableCount_;
};

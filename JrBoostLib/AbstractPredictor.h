#pragma once

class AbstractPredictor {
public:
    virtual ~AbstractPredictor() = default;

    size_t variableCount() const { return variableCount_; }

    ArrayXd predict(CRefXXf inData) const
    { 
        validateInData_(inData);
        size_t sampleCount = static_cast<size_t>(inData.rows());
        pred_.resize(sampleCount);
        pred_ = 0.0;
        predictImpl_(inData, 1.0, pred_);
        return pred_;
    }

protected:
    AbstractPredictor(size_t variableCount) : variableCount_(variableCount) {}

    void validateInData_(CRefXXf inData) const
    {
        const size_t variableCount = inData.cols();
        ASSERT(variableCount == variableCount_);
        ASSERT((inData > -numeric_limits<float>::infinity()).all());
        ASSERT((inData < numeric_limits<float>::infinity()).all());
    }

// deleted:
    AbstractPredictor(const AbstractPredictor&) = delete;
    AbstractPredictor& operator=(const AbstractPredictor&) = delete;

private:
    virtual void predictImpl_(CRefXXf inData, double c, RefXd outData) const = 0;

    size_t variableCount_;

    inline static thread_local ArrayXd pred_;

    friend class BoostTrainer;
    friend class LinearCombinationPredictor;
};

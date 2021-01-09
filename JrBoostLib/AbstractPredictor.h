#pragma once

class AbstractPredictor {
protected:
    AbstractPredictor() = default;

public:
    virtual ~AbstractPredictor() = default;
    virtual size_t variableCount() const = 0;
    virtual ArrayXf predict(const Eigen::ArrayXXf& inData) const = 0;
    
// deleted:
    AbstractPredictor(const AbstractPredictor&) = delete;
    AbstractPredictor& operator=(const AbstractPredictor&) = delete;
    AbstractPredictor(AbstractPredictor&&) = delete;
    AbstractPredictor& operator=(AbstractPredictor&&) = delete;
};

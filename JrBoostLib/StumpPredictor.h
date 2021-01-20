#pragma once

class StumpPredictor {
public:
    StumpPredictor(StumpPredictor&&) = default;
    StumpPredictor& operator=(StumpPredictor&&) = default;
    ~StumpPredictor() = default;

    size_t variableCount() const { return variableCount_; }
    ArrayXd predict(CRefXXf inData) const;

// delete:
    StumpPredictor() = delete;
    StumpPredictor(const StumpPredictor&) = delete;
    StumpPredictor& operator=(const StumpPredictor&) = delete;

private:
    friend class StumpTrainer;

    StumpPredictor(size_t variableCount, size_t j, float x, double leftY, double rightY);
    StumpPredictor(size_t variableCount, double y);

    size_t variableCount_;
    size_t j_;
    float x_;
    double leftY_;
    double rightY_;
};

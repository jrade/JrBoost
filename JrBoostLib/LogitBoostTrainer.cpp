#include "pch.h"
#include "LogitBoostTrainer.h"
#include "LogitBoostOptions.h"

LogitBoostTrainer::LogitBoostTrainer() :
    options_{ std::make_unique<LogitBoostOptions>() }
{}

void LogitBoostTrainer::setInData(CRefXXf inData)
{
    assign(inData_, inData);
    sampleCount_ = inData.rows();
    variableCount_ = inData.cols();
}

void LogitBoostTrainer::setOutData(const ArrayXf& outData)
{
    ASSERT((outData == outData.cast<bool>().cast<float>()).all());	// all elements must be 0 or 1
    outData_ = 2 * outData - 1;
}

void LogitBoostTrainer::setWeights(const ArrayXf& weights)
{
    ASSERT((weights > 0).all());
    ASSERT((weights < std::numeric_limits<float>::infinity()).all());
    weights_ = weights;
}

void LogitBoostTrainer::setOptions(const AbstractOptions& opt)
{
    const LogitBoostOptions& opt1 = dynamic_cast<const LogitBoostOptions&>(opt);
    options_.reset(opt1.clone());
}

BoostPredictor* LogitBoostTrainer::train() const
{
    if (options_->highPrecision())
        return trainImpl_<double>();
    else
        return trainImpl_<float>();
}

template<typename T>
BoostPredictor* LogitBoostTrainer::trainImpl_() const
{
    std::array<T, 2> p{ 0, 0 };
    for (size_t i = 0; i < sampleCount_; ++i)
        p[outData_[i] == 1.0f] += weights_[i];
    T f0 = (std::log(p[1]) - std::log(p[0]));
    Eigen::Array<T, Eigen::Dynamic, 1> F = Eigen::Array<T, Eigen::Dynamic, 1>::Constant(sampleCount_, f0);

    unique_ptr<AbstractTrainer> baseTrainer{ options_->baseOptions()->createTrainer() };
    baseTrainer->setInData(inData_);
    baseTrainer->setOutData(outData_);

    size_t n = options_->iterationCount();
    float eta = options_->eta();
    ArrayXf e2, ce, adjOutData, adjWeights;
    vector<unique_ptr<AbstractPredictor>> basePredictors;

    START_TIMER(t0__);
        
    for (size_t i = 0; i < n; ++i) {

        e2 = (-2 * outData_ * F.cast<float>()).exp();
        ce = (e2  + 1) / 2;  
        adjWeights = weights_ * e2 / ce.square();
        adjOutData = outData_ * ce;

        baseTrainer->setOutData(adjOutData);
        baseTrainer->setWeights(adjWeights);
        unique_ptr<AbstractPredictor> basePredictor{ baseTrainer->train() };
        F += eta * basePredictor->predict(inData_).cast<T>();
        basePredictors.push_back(std::move(basePredictor));
    }

    STOP_TIMER(t0__);
    cout << 1.0e-6 * t0__ << endl;

    return new BoostPredictor(variableCount_, static_cast<float>(f0), eta, std::move(basePredictors));
}

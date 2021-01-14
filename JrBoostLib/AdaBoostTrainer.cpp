#include "pch.h"
#include "AdaBoostTrainer.h"
#include "AdaBoostOptions.h"

AdaBoostTrainer::AdaBoostTrainer() :
    options_{ std::make_unique<AdaBoostOptions>() }
{}

void AdaBoostTrainer::setInData(CRefXXf inData)
{
    assign(inData_, inData);
    sampleCount_ = inData.rows();
    variableCount_ = inData.cols();
}

void AdaBoostTrainer::setOutData(const ArrayXf& outData)
{
    ASSERT((outData == outData.cast<bool>().cast<float>()).all());	// all elements must be 0 or 1
    outData_ = 2 * outData - 1;
}

void AdaBoostTrainer::setWeights(const ArrayXf& weights)
{
    ASSERT((weights > 0).all());
    ASSERT((weights < numeric_limits<float>::infinity()).all());
    weights_ = weights;
}

void AdaBoostTrainer::setOptions(const AbstractOptions& opt)
{
    const AdaBoostOptions& opt1 = dynamic_cast<const AdaBoostOptions&>(opt);
    options_.reset(opt1.clone());
}

BoostPredictor* AdaBoostTrainer::train() const
{
    if (options_->highPrecision())
        return trainImpl_<double>();
    else
        return trainImpl_<float>();
}

template<typename T>
BoostPredictor* AdaBoostTrainer::trainImpl_() const
{
    std::array<T, 2> p{ 0, 0 };
    for (size_t i = 0; i < sampleCount_; ++i)
        p[outData_[i] == 1.0f] += weights_[i];
    T f0 = (std::log(p[1]) - std::log(p[0])) / 2;
    Eigen::Array<T, Eigen::Dynamic, 1> F = Eigen::Array<T, Eigen::Dynamic, 1>::Constant(sampleCount_, f0);

    unique_ptr<AbstractTrainer> baseTrainer{ options_->baseOptions()->createTrainer() };
    baseTrainer->setInData(inData_);
    baseTrainer->setOutData(outData_);

    size_t n = options_->iterationCount();
    float eta = options_->eta();
    ArrayXf adjWeights;
    vector<unique_ptr<AbstractPredictor>> basePredictors;

    START_TIMER(t0__);
        
    for (size_t i = 0; i < n; ++i) {
        adjWeights = weights_ * (-F.cast<float>() * outData_).exp();
        baseTrainer->setWeights(adjWeights);
        unique_ptr<AbstractPredictor> basePredictor{ baseTrainer->train() };
        F += eta * basePredictor->predict(inData_).cast<T>();
        basePredictors.push_back(std::move(basePredictor));
    }

    STOP_TIMER(t0__);
    cout << 1.0e-6 * t0__ << endl;

    return new BoostPredictor(variableCount_, 2 * static_cast<float>(f0), 2 * eta, std::move(basePredictors));
}

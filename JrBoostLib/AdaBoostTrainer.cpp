#include "pch.h"
#include "AdaBoostTrainer.h"
#include "AdaBoostOptions.h"

AdaBoostTrainer::AdaBoostTrainer() :
    options_{ std::make_unique<AdaBoostOptions>() }
{}

void AdaBoostTrainer::setInData(RefXXf inData)
{
    assign(inData_, inData);
    sampleCount_ = inData.rows();
    variableCount_ = inData.cols();
}

void AdaBoostTrainer::setOutData(const ArrayXf& outData)
{
    ASSERT((outData == outData.cast<bool>().cast<float>()).all());	// all elements must be 0 or 1
    outData_ = outData;
}

void AdaBoostTrainer::setWeights(const ArrayXf& weights)
{
    ASSERT(weights.isFinite().all());
    ASSERT((weights > 0).all());
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
    size_t t0 = 0;

    size_t t = clockCycleCount();
    t0 -= t;

    ArrayXf adjOutData{ 2 * outData_ - 1 };
    ArrayXf adjWeights{ sampleCount_ };

    unique_ptr<AbstractOptions> baseOptions{ options_->baseOptions() };
    unique_ptr<AbstractTrainer> baseTrainer{ baseOptions->createTrainer() };
    baseTrainer->setInData(inData_);
    baseTrainer->setOutData(adjOutData);

    std::array<T, 2> p{ 0, 0 };
    for (size_t i = 0; i < sampleCount_; ++i)
        p[adjOutData[i] == 1] += weights_[i];
    T f0 = (log(p[1]) - log(p[0])) / 2;
    Eigen::Array<T, Eigen::Dynamic, 1> F = Eigen::Array<T, Eigen::Dynamic, 1>::Constant(sampleCount_, f0);

    size_t n = options_->iterationCount();
    float eta = options_->eta();
    //float clamp = options_->clamp();

    vector<unique_ptr<AbstractPredictor>> basePredictors;
    for (size_t i = 0; i < n; ++i) {
        //adjWeights = weights_ * (-F.cast<float>() * adjOutData).cwiseMin(clamp).cwiseMax(-clamp).exp();
        adjWeights = weights_ * (-F.cast<float>() * adjOutData).exp();
        baseTrainer->setWeights(adjWeights);
        unique_ptr<AbstractPredictor> basePredictor{ baseTrainer->train() };
        F += eta * basePredictor->predict(inData_).cast<T>();
        basePredictors.push_back(std::move(basePredictor));
    }

    t = clockCycleCount();
    t0 += t;

    cout << 1.0e-6 * t0 << endl;

    return new BoostPredictor(variableCount_, 2 * static_cast<float>(f0), 2 * eta, std::move(basePredictors));
}

#include "pch.h"
#include "AdaBoostTrainer.h"
#include "AdaBoostOptions.h"
#include "StumpTrainer.h"

AdaBoostTrainer::AdaBoostTrainer() :
    options_{ std::make_unique<AdaBoostOptions>() }
{}

void AdaBoostTrainer::setInData(CRefXXf inData)
{
    assign(inData_, inData);
    sampleCount_ = inData.rows();
    variableCount_ = inData.cols();
}

void AdaBoostTrainer::setOutData(const ArrayXd& outData)
{
    ASSERT((outData == outData.cast<bool>().cast<double>()).all());	// all elements must be 0 or 1
    outData_ = 2 * outData - 1;
}

void AdaBoostTrainer::setWeights(const ArrayXd& weights)
{
    ASSERT((weights > 0.0).all());
    ASSERT((weights < numeric_limits<double>::infinity()).all());
    weights_ = weights;
}

void AdaBoostTrainer::setOptions(const AbstractOptions& opt)
{
    const AdaBoostOptions& opt1 = dynamic_cast<const AdaBoostOptions&>(opt);
    options_.reset(opt1.clone());
}

BoostPredictor* AdaBoostTrainer::train() const
{
    const size_t logStep = 1;

    size_t t0 = 0;
    size_t t1 = 0;
    START_TIMER(t0);

    std::array<double, 2> p{ 0.0, 0.0 };
    for (size_t i = 0; i < sampleCount_; ++i)
        p[outData_[i] == 1.0] += weights_[i];
    double f0 = (std::log(p[1]) - std::log(p[0])) / 2.0;
    ArrayXd F = ArrayXd::Constant(sampleCount_, f0);

    unique_ptr<AbstractTrainer> baseTrainer{ options_->baseOptions()->createTrainer() };
    if (StumpTrainer* st = dynamic_cast<StumpTrainer*>(baseTrainer.get()))
        st->setStrata((outData_ == 1.0).cast<size_t>());
    baseTrainer->setInData(inData_);
    baseTrainer->setOutData(outData_);

    const size_t n = options_->iterationCount();
    const double eta = options_->eta();
    ArrayXd adjWeights;
    vector<unique_ptr<AbstractPredictor>> basePredictors;
        
    for (size_t i = 0; i < n; ++i) {
        double FYMin = (F * outData_).minCoeff();
        adjWeights = weights_ * (-F * outData_ + FYMin).exp();
        baseTrainer->setWeights(adjWeights);

        if (i != 0 && i % logStep == 0)
        {
            cout << i << endl;
            cout << "Fy: " << (outData_ * F).minCoeff() << " - " << (outData_ * F).maxCoeff() << endl;
            cout << "w: " << adjWeights.minCoeff() << " - " << adjWeights.maxCoeff();
            cout << " -> " << 100.0 * (adjWeights != 0).cast<double>().sum() / sampleCount_ << "%" << endl;
        }

        SWITCH_TIMER(t0, t1);
        unique_ptr<AbstractPredictor> basePredictor{ baseTrainer->train() };
        SWITCH_TIMER(t1, t0);

        ArrayXd f = basePredictor->predict(inData_);

        if (logStep > 0 && i % logStep == 0)
            cout << "fy: " << (f * outData_).minCoeff() << " - " << (f * outData_).maxCoeff() << endl << endl;

        F += eta * f;

        basePredictors.push_back(std::move(basePredictor));
    }

    STOP_TIMER(t0);
    cout << 1.0e-6 * t0 << endl;
    cout << 1.0e-6 * t1 << endl;
    cout << endl;

    return new BoostPredictor(variableCount_, 2  * f0, vector<double>(n, 2 * eta), std::move(basePredictors));
}

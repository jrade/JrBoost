#include "pch.h"
#include "LogitBoostTrainer.h"
#include "LogitBoostOptions.h"
#include "StumpTrainer.h"

#pragma warning(disable: 4127)

LogitBoostTrainer::LogitBoostTrainer() :
    options_{ std::make_unique<LogitBoostOptions>() }
{}

void LogitBoostTrainer::setInData(CRefXXf inData)
{
    assign(inData_, inData);
    sampleCount_ = inData.rows();
    variableCount_ = inData.cols();
}

void LogitBoostTrainer::setOutData(const ArrayXd& outData)
{
    ASSERT((outData == outData.cast<bool>().cast<double>()).all());	// all elements must be 0 or 1
    outData_ = 2 * outData - 1;
}

void LogitBoostTrainer::setWeights(const ArrayXd& weights)
{
    ASSERT((weights > 0.0).all());
    ASSERT((weights < numeric_limits<double>::infinity()).all());
    weights_ = weights;
}

void LogitBoostTrainer::setOptions(const AbstractOptions& opt)
{
    const LogitBoostOptions& opt1 = dynamic_cast<const LogitBoostOptions&>(opt);
    options_.reset(opt1.clone());
}

BoostPredictor* LogitBoostTrainer::train() const
{
    size_t t0 = 0;
    size_t t1 = 0;
    START_TIMER(t0);

    std::array<double, 2> p{ 0, 0 };
    for (size_t i = 0; i < sampleCount_; ++i)
        p[outData_[i] == 1.0] += weights_[i];
    const double f0 = (std::log(p[1]) - std::log(p[0])) / 2.0;
    ArrayXd F = ArrayXd::Constant(sampleCount_, f0);

    unique_ptr<AbstractTrainer> baseTrainer{ options_->baseOptions()->createTrainer() };
    baseTrainer->setInData(inData_);
    baseTrainer->setOutData(outData_);

    const size_t n = options_->iterationCount();
    const double eta = options_->eta();
    ArrayXd adjOutData, adjWeights;

    vector<double> coeff;
    vector<unique_ptr<AbstractPredictor>> basePredictors;
        
    for (size_t i = 0; i < n; ++i) {

        const size_t logStep = 1;

        if (logStep > 0 && i % logStep == 0) {
            cout << endl << i << endl;
            cout << "Fy: " << (outData_ * F).minCoeff() << " - " << (outData_ * F).maxCoeff() << endl;
        }

        double FAbsMin = F.abs().minCoeff();
        adjWeights = 1.0 / ((F - FAbsMin).exp() + (-F - FAbsMin).exp());
        adjOutData = outData_ * (1.0 + (-2.0 * outData_ * F).exp()) / 2.0;

        if (logStep > 0 && i % logStep == 0) {
            cout << "y*y: " << (outData_ * adjOutData).minCoeff() << " - " << (outData_ * adjOutData).maxCoeff() << endl;
            cout << "w: " << adjWeights.minCoeff() << " - " << adjWeights.maxCoeff();
            cout << " -> " << 100.0 * (adjWeights != 0.0).cast<double>().sum() / sampleCount_ << "%" << endl;
        }

        baseTrainer->setOutData(adjOutData);
        baseTrainer->setWeights(adjWeights);
        if (StumpTrainer* st = dynamic_cast<StumpTrainer*>(baseTrainer.get()))
            st->setStrata((outData_ == 1.0).cast<size_t>());

        SWITCH_TIMER(t0, t1);
        unique_ptr<AbstractPredictor> basePredictor{ baseTrainer->train() };
        SWITCH_TIMER(t1, t0);

        if (logStep != 0 && i % logStep == 0)
            cout << "y*y: " << (outData_ * adjOutData).minCoeff() << " - " << (outData_ * adjOutData).maxCoeff() << endl;

        ArrayXd f = basePredictor->predict(inData_);

        if (logStep > 0 && i % logStep == 0)
            cout << "fy: " << (f * outData_).minCoeff() << " - " << (f * outData_).maxCoeff() << endl << endl;

        double c = eta / std::max(0.5, f.abs().maxCoeff());

        F += c * f;
        coeff.push_back(2 * c);
        basePredictors.push_back(std::move(basePredictor));
    }


    STOP_TIMER(t0);
    cout << 1.0e-6 * t0 << endl;
    cout << 1.0e-6 * t1 << endl;
    cout << endl;

    return new BoostPredictor(variableCount_, 2 * f0, std::move(coeff), std::move(basePredictors));
}

#include "pch.h"
#include "LogitBoostTrainer.h"
#include "LogitBoostOptions.h"
#include "StumpTrainer.h"

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
    ASSERT((weights < numeric_limits<float>::infinity()).all());
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
    size_t t0 = 0;
    size_t t1 = 0;
    START_TIMER(t0);

    std::array<T, 2> p{ 0, 0 };
    for (size_t i = 0; i < sampleCount_; ++i)
        p[outData_[i] == 1.0f] += weights_[i];
    const T f0 = (std::log(p[1]) - std::log(p[0]));
    Eigen::Array<T, Eigen::Dynamic, 1> F = Eigen::Array<T, Eigen::Dynamic, 1>::Constant(sampleCount_, f0);

    unique_ptr<AbstractTrainer> baseTrainer{ options_->baseOptions()->createTrainer() };
    baseTrainer->setInData(inData_);
    baseTrainer->setOutData(outData_);

    const size_t n = options_->iterationCount();
    const float eta = options_->eta();
    ArrayXf e2, ce, adjOutData, adjWeights;
    vector<unique_ptr<AbstractPredictor>> basePredictors;
        
    for (size_t i = 0; i < n; ++i) {

        const size_t logStep = 1;
        if (i % logStep == 0) {
            cout << endl << i << endl;
            cout << "Fy: " << (outData_ * F.cast<float>()).minCoeff() << " - " << (outData_ * F.cast<float>()).maxCoeff() << endl;
        }

        e2 = (-2 * outData_ * F.cast<float>()).exp();
        ce = (1 + e2) / 2;  
        adjWeights = weights_ * e2 / ce.square();
        adjOutData = outData_ * ce;

        if (i % logStep == 0) {
            cout << "w: " << adjWeights.minCoeff() << " - " << adjWeights.maxCoeff();
            cout << " -> " << 100.0f * (adjWeights != 0).cast<float>().sum() / sampleCount_ << "%" << endl;
        }


        baseTrainer->setOutData(adjOutData);
        baseTrainer->setWeights(adjWeights);
        if (StumpTrainer* st = dynamic_cast<StumpTrainer*>(baseTrainer.get()))
            st->setStrata((outData_ == 1.0f).cast<size_t>());

        SWITCH_TIMER(t0, t1);
        unique_ptr<AbstractPredictor> basePredictor{ baseTrainer->train() };
        SWITCH_TIMER(t1, t0);

        if (i % logStep == 0)
            cout << "y*: " << adjOutData.minCoeff() << " - " << adjOutData.maxCoeff() << endl;

        ArrayXf f = basePredictor->predict(inData_);

        if (i % logStep == 0)
            cout << "f: " << f.minCoeff() << " - " << f.maxCoeff() << endl;

        ASSERT(f.isFinite().all());
        F += eta * f.cast<T>();
        basePredictors.push_back(std::move(basePredictor));
    }

/*
    for (size_t i = 0; i < n; ++i) {

        const size_t logStep = 1000;
        if (i % logStep == 0) {
            cout << endl << i << endl;
            cout << "Fy: " << (outData_ * F.cast<float>()).minCoeff() << " - " << (outData_ * F.cast<float>()).maxCoeff() << endl;
        }

        float FYAdj = (outData_ * F.cast<float>()).minCoeff();
        e2 = (-2 * outData_ * F.cast<float>() + 2 * FYAdj).exp();
        ASSERT(e2.isFinite().all());
        ce = (1 + e2 * exp(std::min(-2 * FYAdj, 10.0f))) / 2;
        ASSERT(ce.isFinite().all());
        adjWeights = weights_ * e2 / ce.square();
        adjOutData = outData_ * ce;

        if (i % logStep == 0) {
            cout << "e2: " << e2.minCoeff() << " - " << e2.maxCoeff() << endl;
            cout << "ce: " << ce.minCoeff() << " - " << ce.maxCoeff() << endl;
            cout << "adjW: " << adjWeights.minCoeff() << " - " << adjWeights.maxCoeff();
            cout << " -> " << 100.0f * (adjWeights != 0).cast<float>().sum() / sampleCount_ << "%" << endl;
            cout << "adjY: " << adjOutData.minCoeff() << " - " << adjOutData.maxCoeff() << endl;
        }


        baseTrainer->setOutData(adjOutData);
        baseTrainer->setWeights(adjWeights);

        SWITCH_TIMER(t0, t1);
        unique_ptr<AbstractPredictor> basePredictor{ baseTrainer->train() };
        SWITCH_TIMER(t1, t0);

        ArrayXf f = basePredictor->predict(inData_);
        ASSERT(f.isFinite().all());

        if (i % logStep == 0) {
            cout << "f: " << f.minCoeff() << " - " << f.maxCoeff() << endl;
        }
        F += eta * f.cast<T>();
        basePredictors.push_back(std::move(basePredictor));
    }
*/

    STOP_TIMER(t0);
    cout << 1.0e-6 * t0 << endl;
    cout << 1.0e-6 * t1 << endl;
    cout << endl;

    return new BoostPredictor(variableCount_, static_cast<float>(f0), eta, std::move(basePredictors));
}

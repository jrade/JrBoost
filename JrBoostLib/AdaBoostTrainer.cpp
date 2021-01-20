#include "pch.h"
#include "AdaBoostTrainer.h"
#include "BoostPredictor.h"

AdaBoostTrainer::AdaBoostTrainer(CRefXXf inData, const ArrayXd& outData, const ArrayXd& weights) :
    inData_(inData),
    outData_(2 * outData - 1.0),
    weights_(weights),
    baseTrainer_ { inData, outData }
{
    ASSERT((outData == outData.cast<bool>().cast<double>()).all());	// all elements must be 0 or 1
    ASSERT((weights > 0.0).all());
    ASSERT((weights < numeric_limits<double>::infinity()).all());
}

BoostPredictor AdaBoostTrainer::train(const BoostOptions& opt) const
{
    ASSERT(inData_.rows() != 0);
    ASSERT(inData_.cols() != 0);
    ASSERT(outData_.rows() == inData_.rows());
    ASSERT(weights_.rows() == inData_.rows());

    size_t t0 = 0;
    size_t t1 = 0;
    START_TIMER(t0);

    const size_t sampleCount = inData_.rows();
    const size_t variableCount = inData_.cols();

    double p[2]{ 0.0, 0.0 };
    for (size_t i = 0; i < sampleCount; ++i)
        p[outData_[i] == 1.0] += weights_[i];
    const double f0 = (std::log(p[1]) - std::log(p[0])) / 2.0;
    ArrayXd F = ArrayXd::Constant(sampleCount, f0);

    ArrayXd adjWeights, f;
    vector<StumpPredictor> basePredictors;
    vector<double> coeff;

    const double eta = opt.eta();
    const size_t n = opt.iterationCount();
    for(size_t i = 0; i != n; ++i) {

        double FYMin = (F * outData_).minCoeff();
        adjWeights = weights_ * (-F * outData_ + FYMin).exp();

        SWITCH_TIMER(t0, t1);
        StumpPredictor basePredictor{ baseTrainer_.train(outData_, adjWeights, opt.baseOptions()) };
        SWITCH_TIMER(t1, t0);

        f = basePredictor.predict(inData_);
        F += eta * f;
        basePredictors.push_back(std::move(basePredictor));
        coeff.push_back(2 * eta);

        if (opt.logStep() > 0 && i % opt.logStep() == 0) {
            cout << i << "(" << eta << ")" << endl;
            cout << "Fy: " << (outData_ * F).minCoeff() << " - " << (outData_ * F).maxCoeff() << endl;
            cout << "w: " << adjWeights.minCoeff() << " - " << adjWeights.maxCoeff();
            cout << " -> " << 100.0 * (adjWeights != 0).cast<double>().sum() / sampleCount << "%" << endl;
            cout << "fy: " << (f * outData_).minCoeff() << " - " << (f * outData_).maxCoeff() << endl << endl;
        }
    }

    STOP_TIMER(t0);
    cout << 1.0e-6 * t0 << endl;
    cout << 1.0e-6 * t1 << endl;
    cout << endl;

    return BoostPredictor(variableCount, 2  * f0, std::move(coeff), std::move(basePredictors));
}

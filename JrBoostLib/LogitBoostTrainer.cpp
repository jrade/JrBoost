#include "pch.h"
#include "LogitBoostTrainer.h"
#include "StumpTrainer.h"
#include "BoostPredictor.h"

LogitBoostTrainer::LogitBoostTrainer(CRefXXf inData, const ArrayXd& outData, const ArrayXd& weights) :
    inData_{ inData },
    outData_{ 2.0 * outData - 1.0 },
    weights_{ weights },
    baseTrainer_{ inData, outData }
{
    ASSERT((outData == outData.cast<bool>().cast<double>()).all());	// all elements must be 0 or 1
    ASSERT((weights > 0.0).all());
    ASSERT((weights < numeric_limits<double>::infinity()).all());
}

BoostPredictor LogitBoostTrainer::train(const BoostOptions& opt) const
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

    array<double, 2> p{ 0, 0 };
    for (size_t i = 0; i < sampleCount; ++i)
        p[outData_[i] == 1.0] += weights_[i];
    const double f0 = (std::log(p[1]) - std::log(p[0])) / 2.0;
    ArrayXd F = ArrayXd::Constant(sampleCount, f0);

    ArrayXd adjOutData, adjWeights, f;
    vector<StumpPredictor> basePredictors;
    vector<double> coeff;

    const double eta = opt.eta();
    const size_t n = opt.iterationCount();
    for(size_t i = 0; i < n; ++i) {

        double FAbsMin = F.abs().minCoeff();
        adjWeights = 1.0 / ((F - FAbsMin).exp() + (-F - FAbsMin).exp()).square();
        adjWeights /= adjWeights.maxCoeff();

        adjOutData = outData_ * (1.0 + (-2.0 * outData_ * F).exp()) / 2.0;

        SWITCH_TIMER(t0, t1);
        StumpPredictor basePredictor{ baseTrainer_.train(adjOutData, adjWeights, opt.baseOptions()) };
        SWITCH_TIMER(t1, t0);

        f = basePredictor.predict(inData_);
        double c = eta / std::max(0.5, f.abs().maxCoeff());
        F += c * f;
        coeff.push_back(2 * c);
        basePredictors.push_back(std::move(basePredictor));

        if (opt.logStep() > 0 && i % opt.logStep() == 0) {
            cout << i << "(" << eta << ")" << endl;
            cout << "Fy: " << (outData_ * F).minCoeff() << " - " << (outData_ * F).maxCoeff() << endl;
            cout << "w: " << adjWeights.minCoeff() << " - " << adjWeights.maxCoeff();
            cout << " -> " << 100.0 * (adjWeights != 0.0).cast<double>().sum() / sampleCount << "%" << endl;
            cout << "y*y: " << (outData_ * adjOutData).minCoeff() << " - " << (outData_ * adjOutData).maxCoeff() << endl;
            cout << "fy: " << (f * outData_).minCoeff() << " - " << (f * outData_).maxCoeff() << endl << endl;
        }
    }

    STOP_TIMER(t0);
    cout << 1.0e-6 * t0 << endl;
    cout << 1.0e-6 * t1 << endl;
    cout << endl;

    return BoostPredictor(variableCount, 2 * f0, std::move(coeff), std::move(basePredictors));
}

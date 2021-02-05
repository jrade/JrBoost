#include "pch.h"
#include "BoostTrainer.h"
#include "BoostOptions.h"
#include "LinearCombinationPredictor.h"
#include "StumpTrainer.h"
#include "Util.h"


BoostTrainer::BoostTrainer(CRefXXf inData, RefXs outData) :
    inData_{ inData },
    rawOutData_{ outData },
    outData_{ 2.0 * rawOutData_.cast<double>() - 1.0 },
    f0_{ (
        log(static_cast<double>(outData.sum()))
            - log(static_cast<double>((1 - outData).sum()))
        ) / 2.0 }
{
    ASSERT(inData_.rows() != 0);
    ASSERT(inData_.cols() != 0);
    ASSERT(outData_.rows() == inData_.rows());

    ASSERT((inData > -numeric_limits<float>::infinity()).all());
    ASSERT((inData < numeric_limits<float>::infinity()).all());
    ASSERT((outData < 2).all());
}

unique_ptr<AbstractPredictor> BoostTrainer::train(const BoostOptions& opt) const
{
    CLOCK::PUSH(CLOCK::BT_TRAIN);
    auto pred = opt.method() == BoostOptions::Method::Ada ? trainAda_(opt) : trainLogit_(opt);
    size_t sampleCount = static_cast<size_t>(inData_.rows());
    size_t iterationCount = opt.iterationCount();
    CLOCK::POP(sampleCount * iterationCount);
    return pred;
}

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<AbstractPredictor> BoostTrainer::trainAda_(const BoostOptions& opt) const
{
    const size_t sampleCount = inData_.rows();
    const size_t variableCount = inData_.cols();
    const double eta = opt.eta();
    const size_t iterationCount = opt.iterationCount();

    ArrayXd F(sampleCount);
    ArrayXd Fy(sampleCount);
    ArrayXd adjWeights(sampleCount);

    F = f0_;

    vector<unique_ptr<AbstractPredictor>> basePredictors(iterationCount);
    vector<double> coeff(iterationCount);

    for (size_t i = 0; i < iterationCount; ++i) {

        Fy = F * outData_;
        adjWeights = (-Fy + Fy.minCoeff()).exp();
        unique_ptr<AbstractPredictor> basePred = baseTrainer_->train(outData_, adjWeights, opt.base());
        basePred->predict(inData_, eta, F);

        basePredictors[i] = std::move(basePred);
        coeff[i] = 2 * eta;

        if (opt.logStep() > 0 && i % opt.logStep() == 0) {
            cout << i << "(" << eta << ")" << endl;
            cout << "Fy: " << Fy.minCoeff() << " - " << Fy.maxCoeff() << endl;
            cout << "w: " << adjWeights.minCoeff() << " - " << adjWeights.maxCoeff();
            cout << " -> " << 100.0 * (adjWeights != 0).cast<double>().sum() / sampleCount << "%" << endl;
        }
    }

    return std::make_unique<LinearCombinationPredictor>(variableCount, 2 * f0_, std::move(coeff), std::move(basePredictors));
}

//----------------------------------------------------------------------------------------------------------------------

unique_ptr<AbstractPredictor> BoostTrainer::trainLogit_(const BoostOptions& opt) const
{
    const size_t sampleCount = inData_.rows();
    const size_t variableCount = inData_.cols();

    ArrayXd F = ArrayXd::Constant(sampleCount, f0_);

    ArrayXd adjOutData, adjWeights, f;
    vector<unique_ptr<AbstractPredictor>> basePredictors;
    vector<double> coeff;

    const double eta = opt.eta();
    const size_t n = opt.iterationCount();
    for (size_t i = 0; i < n; ++i) {

        double FAbsMin = F.abs().minCoeff();
        adjWeights = 1.0 / ((F - FAbsMin).exp() + (-F - FAbsMin).exp()).square();
        adjWeights /= adjWeights.maxCoeff();

        adjOutData = outData_ * (1.0 + (-2.0 * outData_ * F).exp()) / 2.0;

        unique_ptr<AbstractPredictor > basePredictor = baseTrainer_->train(adjOutData, adjWeights, opt.base());

        f = basePredictor->predict(inData_);
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

    return std::make_unique<LinearCombinationPredictor>(variableCount, 2 * f0_, std::move(coeff), std::move(basePredictors));
}

//----------------------------------------------------------------------------------------------------------------------

ArrayXd BoostTrainer::trainAndEval(CRefXXf testInData, CRefXs testOutData, const vector<BoostOptions>& opt) const
{
    ASSERT(testInData.rows() == testOutData.rows());
    size_t testSampleCount = static_cast<size_t>(testInData.rows());
    size_t optCount = opt.size();

    // In each iteration of the loop we build and evaluate a classifier with one set of options
    // Differents sets of options can take very different time.
    // To ensure that the OMP threads are balanced we use dynamical scheduling and we sort the options objects
    // from the most time-consuming to the least.

    vector<size_t> optIndicesSortedByCost(optCount);
    sortedIndices(
        cbegin(opt), 
        cend(opt),
        begin(optIndicesSortedByCost),
        [](const auto& opt) { return -opt.cost(); }
    );

    ArrayXd scores(optCount);
    std::exception_ptr ep;

    #pragma omp parallel
    {
        ArrayXd predData(testSampleCount);

        #pragma omp for nowait schedule(dynamic)
        for (int i = 0; i < static_cast<int>(optCount); ++i) {
            try {
                size_t j = optIndicesSortedByCost[i];
                unique_ptr<AbstractPredictor> pred = train(opt[j]);
                predData = 0.0;
                pred->predict(testInData, 1.0, predData);
                scores(j) = linLoss(testOutData, predData);
            }
            catch (const std::exception&) {
                #pragma omp critical
                if (!ep) ep = std::current_exception();
            }
        } // don't wait here ...
        CLOCK::PUSH(CLOCK::OMP_BARRIER);
    } // ... but here so we can measure the wait time
    CLOCK::POP();

    if (ep) std::rethrow_exception(ep);
    return scores;
}

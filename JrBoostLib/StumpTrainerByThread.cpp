#include "pch.h"
#include "StumpTrainerByThread.h"
#include "StumpTrainerShared.h"
#include "StumpOptions.h"
#include "StumpPredictor.h"
#include "TrivialPredictor.h"
#include "FastAlgorithms.h"


StumpTrainerByThread::StumpTrainerByThread(
    CRefXXf inData,
    shared_ptr<const StumpTrainerShared> shared,
    std::random_device& rd
) :
    inData_(inData),
    sampleCount_(inData.rows()),
    variableCount_(inData.cols()),
    shared_( shared ),
    rne_{ rd }
{
}


unique_ptr<AbstractPredictor> StumpTrainerByThread::train(CRefXd outData, CRefXd weights, const StumpOptions& options)
{
    CLOCK::PUSH(CLOCK::USED_SAMPLES);
    size_t usedSampleCount = shared_->initUsedSampleMask(&usedSampleMask_, options, rne_);
    CLOCK::POP(sampleCount_);

    CLOCK::PUSH(CLOCK::USED_VARIABLES);
    size_t candidateVariableCount = initUsedVariables_(options);
    CLOCK::POP(candidateVariableCount);

    CLOCK::PUSH(CLOCK::SUMS);
    initSums_(outData, weights);
    CLOCK::POP(usedSampleCount);

    if (sumW_ == 0) {
        cout << "Warning: sumW = 0" << endl;
        return std::make_unique<TrivialPredictor>(variableCount_, 0.0);
    }

    double bestScore = sumWY_ * sumWY_ / sumW_;
    size_t bestJ = static_cast<size_t>(-1);
    float bestX = 0.0f;
    double bestLeftY = 0.0;
    double bestRightY = 0.0;

    const size_t minNodeSize = options.minNodeSize();
    const double tol = sumW_ * sqrt(static_cast<double>(usedSampleCount)) * numeric_limits<double>::epsilon() / 2;
    // tol = estimate of the rounding off error we can expect in rightSumW towards the end of the loop
    const double minNodeWeight = std::max<double>(options.minNodeWeight(), tol);
        
    for (size_t j : usedVariables_) {

        CLOCK::PUSH(CLOCK::SORTED_USED_SAMPLES);
        shared_->initSortedUsedSamples(&sortedUsedSamples_, usedSampleCount, usedSampleMask_, j);
        CLOCK::POP(sampleCount_);

        // find best split

        double leftSumW = 0.0;
        double leftSumWY = 0.0;
        double rightSumW = sumW_;
        double rightSumWY = sumWY_;

        const auto pBegin = cbegin(sortedUsedSamples_);
        const auto pEnd = cend(sortedUsedSamples_);
        auto p = pBegin;
        size_t nextI = *p;

        // this is where most execution time is spent ......

        CLOCK::PUSH(CLOCK::BEST_SPLIT);
            
        while (p != pEnd - 1) {

            const size_t i = nextI;
            nextI = *++p;
            const double w = weights[i];
            const double y = outData[i];
            leftSumW += w;
            rightSumW -= w;
            leftSumWY += w * y;
            rightSumWY -= w * y;
            const double score = leftSumWY * leftSumWY / leftSumW + rightSumWY * rightSumWY / rightSumW;

            if (score <= bestScore) continue;  // usually true

            ++CLOCK::slowBranchCount;

        //..................................................

            if (p < pBegin + minNodeSize
                || p > pEnd - minNodeSize
                || leftSumW < minNodeWeight
                || rightSumW < minNodeWeight
            ) continue;

            const float leftX = inData_(i, j);
            const float rightX = inData_(nextI, j);
            const float midX = (leftX + rightX) / 2;
    
            if (leftX == midX) continue;

            bestScore = score;
            bestJ = j;
            bestX = midX;
            bestLeftY = leftSumWY / leftSumW;
            bestRightY = rightSumWY / rightSumW;
        }

        CLOCK::POP(usedSampleCount);  // CLOCK::BEST_SPLIT
        CLOCK::splitIterationCount += usedSampleCount;
    }

    //usedSampleMask_.shrink_to_fit();
    //usedVariables_.shrink_to_fit();
    //sortedUsedSamples_.shrink_to_fit();

    //vector<char>().swap(usedSampleMask_);
    //vector<size_t>().swap(usedVariables_);
    //vector<SampleIndex>().swap(sortedUsedSamples_);

    if (bestJ == static_cast<size_t>(-1))
        return std::make_unique<TrivialPredictor>(variableCount_, sumWY_ / sumW_);
    else
        return unique_ptr<StumpPredictor>(new StumpPredictor(variableCount_, bestJ, bestX, bestLeftY, bestRightY));
};


size_t StumpTrainerByThread::initUsedVariables_(const StumpOptions& options)
{
    size_t candidateVariableCount = std::min(variableCount_, options.topVariableCount());

    size_t usedVariableCount = std::max(
        static_cast<size_t>(1),
        static_cast<size_t>(static_cast<double>(options.usedVariableRatio()) * candidateVariableCount + 0.5)
    );

    ASSERT(0 < usedVariableCount);
    ASSERT(usedVariableCount <= candidateVariableCount);
    ASSERT(candidateVariableCount <= variableCount_);

    usedVariables_.resize(usedVariableCount);
    auto q0 = begin(usedVariables_);
    auto q1 = end(usedVariables_);
    size_t n = candidateVariableCount;
    fastRandomSubIndices(n, q0, q1, rne_);

    return candidateVariableCount;
}


void StumpTrainerByThread::initSums_(const CRefXd& outData, const CRefXd& weights)
{
    sumW_ = 0.0;
    sumWY_ = 0.0;
    for (size_t i = 0; i < sampleCount_; ++i) {
        double m = usedSampleMask_[i];
        double w = weights[i];
        double y = outData[i];
        sumW_ += m * w;
        sumWY_ += m * w * y;
    }
}

#include "pch.h"
#include "StumpTrainerByThread.h"
#include "StumpTrainerShared.h"
#include "StumpOptions.h"
#include "StumpPredictor.h"
#include "TrivialPredictor.h"


StumpTrainerByThread::StumpTrainerByThread(CRefXXf inData, shared_ptr<const StumpTrainerShared> shared, std::random_device& rd) :
    inData_(inData),
    sampleCount_(inData.rows()),
    variableCount_(inData.cols()),
    shared_( shared ),
    rne_{ rd }
{
}


unique_ptr<AbstractPredictor> StumpTrainerByThread::train(CRefXd outData, CRefXd weights, const StumpOptions& options)
{
    size_t n = 0;
    size_t t0 = 0;
    size_t t1 = 0;
    size_t t2 = 0;

    START_TIMER(t0);
    size_t usedSampleCount = shared_->initUsedSampleMask(&usedSampleMask_, options, rne_);
    initUsedVariables_(options);
    initSums_(outData, weights);
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
        
    SWITCH_TIMER(t0, t1);

    for (size_t j : usedVariables_) {

        shared_->initSortedUsedSamples(&sortedUsedSamples_, usedSampleCount, usedSampleMask_, j);

        // find best split
        SWITCH_TIMER(t1, t2);

        double leftSumW = 0.0;
        double leftSumWY = 0.0;
        double rightSumW = sumW_;
        double rightSumWY = sumWY_;

        const auto pBegin = cbegin(sortedUsedSamples_);
        const auto pEnd = cend(sortedUsedSamples_);
        auto p = pBegin;
        size_t nextI = *p;

        // this is where most execution time is spent ......

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

        //..................................................

            ++n;

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

        //{
        //    const double w = weights_[nextI];
        //    const double y = outData_[nextI];
        //    leftSumW += w;
        //    rightSumW -= w;
        //    leftSumWY += w * y;
        //    rightSumWY -= w * y;
        //}

        SWITCH_TIMER(t2, t1);
    }

    STOP_TIMER(t1);

    if (options.profile()) {
        size_t usedVariableCount = usedVariables_.size();
        cout << t0 << endl;
        cout << t1 << " (" << static_cast<float>(t1) / (sampleCount_ * usedVariableCount) << ")" << endl;
        cout << t2 << " (" << static_cast<float>(t2) / (usedSampleCount * usedVariableCount) << ")" << endl;
        cout << 100.0 * n / ((usedSampleCount - 1) * usedVariableCount) << "%" << endl;
        cout << endl;
    }

    if (bestJ == static_cast<size_t>(-1))
        return std::make_unique<TrivialPredictor>(variableCount_, sumWY_ / sumW_);
    return unique_ptr<AbstractPredictor>(new StumpPredictor(variableCount_, bestJ, bestX, bestLeftY, bestRightY));
};


void StumpTrainerByThread::initUsedVariables_(const StumpOptions& options)
{
    size_t candidateVariableCount = variableCount_;
    if (options.topVariableCount())
        candidateVariableCount = std::min(candidateVariableCount, *options.topVariableCount());

    size_t usedVariableCount = std::max(
        static_cast<size_t>(1),
        static_cast<size_t>(static_cast<double>(options.usedVariableRatio()) * candidateVariableCount + 0.5)
    );

    ASSERT(0 < usedVariableCount);
    ASSERT(usedVariableCount <= candidateVariableCount);
    ASSERT(candidateVariableCount <= variableCount_);

    // now select usedVariableCount variables at random among the first candidateVariableCount variables

    usedVariables_.resize(candidateVariableCount);
    std::iota(begin(usedVariables_), end(usedVariables_), 0);

    fastOrderedRandomSubset(
        cbegin(usedVariables_),
        cend(usedVariables_),
        begin(usedVariables_),
        begin(usedVariables_) + usedVariableCount,
        rne_
    );
    usedVariables_.resize(usedVariableCount);
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

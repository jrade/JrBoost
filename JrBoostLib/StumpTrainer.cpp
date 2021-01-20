#include "pch.h"
#include "StumpTrainer.h"


StumpTrainer::StumpTrainer(CRefXXf inData, const ArrayXd& strata) :
    inData_{ inData },
    sampleCount_{ static_cast<size_t>(inData.rows()) },
    variableCount_{ static_cast<size_t>(inData.cols()) },
    strata_{ strata.cast<size_t>() },
    stratum0Count_{ (strata == 0.0).cast<size_t>().sum() },
    stratum1Count_{ (strata == 1.0).cast<size_t>().sum() }
{
    ASSERT(inData.isFinite().all());
    ASSERT(inData.rows() > 0);
    ASSERT(inData.cols() > 0);
    ASSERT((strata == strata.cast<bool>().cast<double>()).all());	// all elements must be 0 or 1
    ASSERT(inData.rows() == strata.rows());

    sortedSamples_.resize(variableCount_);
    vector<std::pair<float, size_t>> tmp{ sampleCount_ };
    for (size_t j = 0; j < variableCount_; ++j) {
        for (size_t i = 0; i < sampleCount_; ++i)
            tmp[i] = { inData(i,j), i };
        std::sort(begin(tmp), end(tmp));
        sortedSamples_[j].resize(sampleCount_);
        for (size_t i = 0; i < sampleCount_; ++i)
            sortedSamples_[j][i] = tmp[i].second;
    }
}

StumpPredictor StumpTrainer::train(const ArrayXd& outData, const ArrayXd& weights, const StumpOptions& options) const
{
    ASSERT(static_cast<size_t>(outData.rows()) == sampleCount_);
    ASSERT(outData.isFinite().all());

    ASSERT(static_cast<size_t>(weights.rows()) == sampleCount_);
    ASSERT((weights >= 0.0).all());
    ASSERT((weights < numeric_limits<double>::infinity()).all());

    size_t n = 0;
    size_t t0 = 0;
    size_t t1 = 0;
    size_t t2 = 0;
    START_TIMER(t0);

    // used sample mask

    size_t usedSampleCount;
    usedSampleMask_.resize(sampleCount_);

    if (options.isStratified()) {

        array<size_t, 2> sampleCountByStratum{ stratum0Count_, stratum1Count_ };

        array<size_t, 2> usedSampleCountByStratum{
            std::max(
                static_cast<size_t>(1),
                static_cast<size_t>(static_cast<double>(options.usedSampleRatio()) * stratum0Count_ + 0.5)
            ),
            std::max(
                static_cast<size_t>(1),
                static_cast<size_t>(static_cast<double>(options.usedSampleRatio()) * stratum1Count_ + 0.5)
            )
        };

        usedSampleCount = usedSampleCountByStratum[0] + usedSampleCountByStratum[1];

        //cout << sampleCountByStratum[0] << " " << sampleCountByStratum[1] << " ";
        //cout << usedSampleCountByStratum[0] << " " << usedSampleCountByStratum[1] << endl;

        fastStratifiedRandomMask(
            &strata_(0),
            &strata_(0) + sampleCount_,
            begin(usedSampleMask_),
            begin(sampleCountByStratum),
            begin(usedSampleCountByStratum),
            fastRNE_
        );

        //ASSERT(sampleCountByStratum[0] == 0);
        //ASSERT(sampleCountByStratum[1] == 0);
        //ASSERT(usedSampleCountByStratum[0] == 0);
        //ASSERT(usedSampleCountByStratum[1] == 0);
    }
    
    else {
        usedSampleCount = std::max(
            static_cast<size_t>(1),
            static_cast<size_t>(static_cast<double>(options.usedSampleRatio()) * sampleCount_ + 0.5)
        );

        fastRandomMask(
            begin(usedSampleMask_), 
            end(usedSampleMask_), 
            usedSampleCount, 
            fastRNE_
        );
    }

    // used variables

    size_t usedVariableCount = std::max(
        static_cast<size_t>(1),
        static_cast<size_t>(static_cast<double>(options.usedVariableRatio()) * variableCount_ + 0.5)
    );

    usedVariables_.resize(variableCount_);
    std::iota(begin(usedVariables_), end(usedVariables_), 0);
    fastOrderedRandomSubset(
        begin(usedVariables_), 
        end(usedVariables_), 
        begin(usedVariables_), 
        begin(usedVariables_) + usedVariableCount, 
        fastRNE_
    );
 
    usedVariables_.resize(usedVariableCount);

    // sums

    double sumW = 0.0;
    double sumWY = 0.0;
    for (size_t i = 0; i < sampleCount_; ++i) {
        double m = usedSampleMask_[i];
        double w = weights[i];
        double y = outData[i];
        sumW += m * w;
        sumWY += m * w * y;
    }

    ASSERT(sumW != 0);

    // find best split

    double bestScore = sumWY * sumWY / sumW;
    size_t bestJ = static_cast<size_t>(-1);
    float bestX = numeric_limits<float>::quiet_NaN();
    double bestLeftY = numeric_limits<double>::quiet_NaN();
    double bestRightY = numeric_limits<double>::quiet_NaN();

    sortedUsedSamples_.resize(usedSampleCount);
    size_t minNodeSize = options.minNodeSize();
    const double tol = sumW * sqrt(static_cast<double>(usedSampleCount)) * numeric_limits<double>::epsilon() / 2;
    // tol = estimate of the rounding off error we can expect in rightSumW towards the end of the loop
    const double minNodeWeight = std::max<double>(options.minNodeWeight(), tol);
        
    for (size_t j : usedVariables_) {

        // prepare list of samples
        SWITCH_TIMER(t0, t1);

        fastCopyIf(
            begin(sortedSamples_[j]),
            begin(sortedUsedSamples_),
            end(sortedUsedSamples_),
            [&](size_t i) { return usedSampleMask_[i]; }
        );

        // find best split
        SWITCH_TIMER(t1, t2);

        double leftSumW = 0.0;
        double leftSumWY = 0.0;
        double rightSumW = sumW;
        double rightSumWY = sumWY;

        const auto pBegin = begin(sortedUsedSamples_);
        const auto pEnd = end(sortedUsedSamples_);
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

        SWITCH_TIMER(t2, t0);
    }

    STOP_TIMER(t0);

    if (options.profile()) {
        cout << t0 << endl;
        cout << t1 << " (" << static_cast<float>(t1) / (sampleCount_ * usedVariableCount) << ")" << endl;
        cout << t2 << " (" << static_cast<float>(t2) / (usedSampleCount * usedVariableCount) << ")" << endl;
        cout << 100.0 * n / ((usedSampleCount - 1) * usedVariableCount) << "%" << endl;
        cout << endl;
    }

    if (bestJ == static_cast<size_t>(-1))
        return StumpPredictor{ variableCount_, sumWY / sumW };
    else
        return StumpPredictor{ 
            variableCount_, bestJ, bestX, bestLeftY, bestRightY };
};

#pragma once

#include "pdqsort.h"


inline Array3d errorCount_(CRefXs outData, CRefXd predData, double threshold)
{
    double falsePos = static_cast<double>(((1 - outData) * (predData >= threshold).cast<size_t>()).sum());
    double falseNeg = static_cast<double>((outData * (predData < threshold).cast<size_t>()).sum());
    return { falsePos, falseNeg, falsePos + falseNeg };
}

inline Array3d errorCount_lor(CRefXs outData, CRefXd predData)
{
    return errorCount_(outData, predData, 0.0);
}

inline Array3d errorCount_p(CRefXs outData, CRefXd predData)
{
    return errorCount_(outData, predData, 0.5);
}


inline Array3d linLoss_lor(CRefXs outData, CRefXd predData)
{
    double falsePos = ((1 - outData).cast<double>() / (1.0 + (-predData).exp())).sum();
    double falseNeg = (outData.cast<double>() / (1.0 + predData.exp())).sum();
    return { falsePos, falseNeg, falsePos + falseNeg };
}


inline Array3d linLoss_p(CRefXs outData, CRefXd predData)
{
    double falsePos = ((1 - outData).cast<double>() * predData).sum();
    double falseNeg = (outData.cast<double>() * (1.0 - predData)).sum();
    return { falsePos, falseNeg, falsePos + falseNeg };
}


inline Array3d logLoss_lor(CRefXs outData, CRefXd predData)
{
    double falsePos = -(
        (1 - outData).cast<double>()
        * (predData >= 0.0).select(
            -predData - (-predData).exp().log1p()
            , -(predData).exp().log1p()
        )
    ).sum();

    // The two expressions  -predData - (-predData).exp().log1p()  and  -(predData).exp().log1p()
    // are mathematically equivalent but the first is numerically accurate for large positive predData
    // and the second for large negative predData

    double falseNeg = -(outData.cast<double>()
        * (predData >= 0.0).select(
            -(-predData).exp().log1p()
            , predData - (predData).exp().log1p()
        )
    ).sum();

    return { falsePos, falseNeg, falsePos + falseNeg };
}


inline Array3d logLoss_p(CRefXs outData, CRefXd predData)
{
    constexpr double epsilon = std::numeric_limits<double>::min();
    double falsePos = -((1 - outData).cast<double>() * (1.0 - predData + epsilon).log()).sum();
    double falseNeg = -(outData.cast<double>() * (predData + epsilon).log()).sum();
    return { falsePos, falseNeg, falsePos + falseNeg };
}


inline Array3d alphaLoss_lor(CRefXs outData, CRefXd predData, double alpha)
{
    double falsePos = ((1 - outData).cast<double>()
        * (1.0 - (1.0 / (1.0 + predData.exp())).pow(alpha))
        ).sum() / alpha;

    double falseNeg = (outData.cast<double>()
        * (1.0 - (1.0 / (1.0 + (-predData).exp())).pow(alpha))
        ).sum() / alpha;

    return { falsePos, falseNeg, falsePos + falseNeg };
}

inline Array3d alphaLoss_p(CRefXs outData, CRefXd predData, double alpha)
{
    double falsePos = ((1 - outData).cast<double>() * (1.0 - (1.0 - predData).pow(alpha))).sum() / alpha;
    double falseNeg = (outData.cast<double>() * (1.0 - predData.pow(alpha))).sum() / alpha;
    return { falsePos, falseNeg, falsePos + falseNeg };
}


inline Array3d sqrtLoss_lor(CRefXs outData, CRefXd predData)
{
    return alphaLoss_lor(outData, predData, 0.5);
}


inline Array3d sqrtLoss_p(CRefXs outData, CRefXd predData)
{
    return alphaLoss_p(outData, predData, 0.5);
}


inline Array3d negAuc(CRefXs outData, CRefXd predData)
{
    size_t sampleCount = outData.rows();

    vector<pair<size_t, double>> tmp(sampleCount);
    for (size_t i = 0; i < sampleCount; ++i)
        tmp[i] = { outData[i], predData[i] };
    pdqsort_branchless(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return x.second < y.second; });

    size_t a = 0;
    size_t b = 0;
    for (auto [y, pred] : tmp) {
        //if (y == 0)
        //    a += 1;
        //else
        //    b += a;
        a += 1 - y;
        b += y * a;
    }
    double auc = static_cast<double>(b) / (static_cast<double>(a) * static_cast<double>(sampleCount - a));
    const double nan = numeric_limits<double>::quiet_NaN();
    return { nan, nan, -auc };
}

//----------------------------------------------------------------------------------------------------------------------

class ErrorCount {
public:
    ErrorCount(double threshold) : threshold_(threshold) {}

    Array3d operator()(CRefXs outData, CRefXd predData) const
    {
        return errorCount_(outData, predData, threshold_);
    }

    string name() const
    {
        stringstream ss;
        ss << "err-count(" << threshold_ << ")";
        return ss.str();
    }

private:
    const double threshold_;
};


class AlphaLoss_lor {
public:
    AlphaLoss_lor(double alpha) : alpha_(alpha) {}

    Array3d operator()(CRefXs outData, CRefXd predData) const
    {
        return alphaLoss_lor(outData, predData, alpha_);
    }

    string name() const
    {
        stringstream ss;
        ss << "alpha-loss(" << alpha_ << ")";
        return ss.str();
    }

private:
    const double alpha_;
};


class AlphaLoss_p {
public:
    AlphaLoss_p(double alpha) : alpha_(alpha) {}

    Array3d operator()(CRefXs outData, CRefXd predData) const
    {
        return alphaLoss_p(outData, predData, alpha_);
    }

    string name() const
    {
        stringstream ss;
        ss << "alpha-loss(" << alpha_ << ")";
        return ss.str();
    }

private:
    const double alpha_;
};

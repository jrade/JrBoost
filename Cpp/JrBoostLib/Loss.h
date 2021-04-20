//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

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


inline Array3d logLoss_lor(CRefXs outData, CRefXd predData, double gamma)
{
    ASSERT(gamma > 0.0);

    double falsePos = ((1 - outData).cast<double>()
        * (1.0 - (1.0 / (1.0 + predData.exp())).pow(gamma))
        ).sum() / gamma;

    double falseNeg = (outData.cast<double>()
        * (1.0 - (1.0 / (1.0 + (-predData).exp())).pow(gamma))
        ).sum() / gamma;

    return { falsePos, falseNeg, falsePos + falseNeg };
}

inline Array3d logLoss_p(CRefXs outData, CRefXd predData, double gamma)
{
    ASSERT(gamma > 0.0);

    double falsePos = ((1 - outData).cast<double>() * (1.0 - (1.0 - predData).pow(gamma))).sum() / gamma;
    double falseNeg = (outData.cast<double>() * (1.0 - predData.pow(gamma))).sum() / gamma;
    return { falsePos, falseNeg, falsePos + falseNeg };
}


inline Array3d auc(CRefXs outData, CRefXd predData)
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
    return { nan, nan, auc };
}


inline Array3d negAuc(CRefXs outData, CRefXd predData)
{
    return -auc(outData, predData);
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


class LogLoss_lor {
public:
    LogLoss_lor(double gamma) : gamma_(gamma)
    {
        ASSERT(gamma > 0.0);
    }

    Array3d operator()(CRefXs outData, CRefXd predData) const
    {
        return logLoss_lor(outData, predData, gamma_);
    }

    string name() const
    {
        stringstream ss;
        ss << "gamma(" << gamma_ << ")";
        return ss.str();
    }

private:
    const double gamma_;
};


class LogLoss_p {
public:
    LogLoss_p(double gamma) : gamma_(gamma)
    {
        ASSERT(gamma > 0.0);
    }

    Array3d operator()(CRefXs outData, CRefXd predData) const
    {
        return logLoss_p(outData, predData, gamma_);
    }

    string name() const
    {
        stringstream ss;
        ss << "gamma(" << gamma_ << ")";
        return ss.str();
    }

private:
    const double gamma_;
};

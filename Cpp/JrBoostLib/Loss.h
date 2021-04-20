//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "pdqsort.h"


inline Array3d errorCount(CRefXs outData, CRefXd predData, double threshold = 0.5)
{
    double falsePos = static_cast<double>(((1 - outData) * (predData >= threshold).cast<size_t>()).sum());
    double falseNeg = static_cast<double>((outData * (predData < threshold).cast<size_t>()).sum());
    return { falsePos, falseNeg, falsePos + falseNeg };
}

inline Array3d linLoss(CRefXs outData, CRefXd predData)
{
    double falsePos = ((1 - outData).cast<double>() * predData).sum();
    double falseNeg = (outData.cast<double>() * (1.0 - predData)).sum();
    return { falsePos, falseNeg, falsePos + falseNeg };
}

inline Array3d logLoss(CRefXs outData, CRefXd predData, double gamma = 0.1)
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
        return errorCount(outData, predData, threshold_);
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


class LogLoss {
public:
    LogLoss(double gamma) : gamma_(gamma)
    {
        ASSERT(gamma > 0.0);
    }

    Array3d operator()(CRefXs outData, CRefXd predData) const
    {
        return logLoss(outData, predData, gamma_);
    }

    string name() const
    {
        stringstream ss;
        ss << "log-loss(" << gamma_ << ")";
        return ss.str();
    }

private:
    const double gamma_;
};

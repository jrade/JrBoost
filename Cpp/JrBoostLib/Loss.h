//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "pdqsort.h"


inline void lossFunValidate_(CRefXs outData, CRefXd predData)
{
    if (outData.rows() != predData.rows())
        throw std::invalid_argument("True outdata and predicted outdata have different numbers of samples.");
    if ((outData > 1).any())
        throw std::invalid_argument("True outdata has values that are not 0 or 1.");
    if (!(predData >= 0.0f && predData <= 1.0f).all())
        throw std::invalid_argument("Predicted outdata has values that do not lie in the interval [0.0, 1.0].");
}


inline Array3d errorCount(CRefXs outData, CRefXd predData, double threshold = 0.5)
{
    lossFunValidate_(outData, predData);
    if (!(threshold > 0 && threshold < 1.0))
        throw std::invalid_argument("threshold must lie in the interval (0.0, 1.0).");

    const size_t falsePos = (((1 - outData) * (predData >= threshold).cast<size_t>()).sum());
    const size_t falseNeg = ((outData * (predData < threshold).cast<size_t>()).sum());

    return { 
        static_cast<double>(falsePos),
        static_cast<double>(falseNeg),
        static_cast<double>(falsePos + falseNeg)
    };
}

inline Array3d senseSpec(CRefXs outData, CRefXd predData, double threshold = 0.5)
{
    lossFunValidate_(outData, predData);
    if (!(threshold > 0 && threshold < 1.0))
        throw std::invalid_argument("threshold must lie in the interval (0.0, 1.0).");

    const  size_t pos = outData.sum();
    const size_t neg = (1 - outData).sum();
    const size_t truePos = (outData * (predData >= threshold).cast<size_t>()).sum();
    const size_t trueNeg = ((1 - outData) * (predData < threshold).cast<size_t>()).sum();

    return {
        static_cast<double>(truePos) / pos,
        static_cast<double>(trueNeg) / neg,
        static_cast<double>(truePos + trueNeg) / (pos + neg)
    };
}

inline Array3d linLoss(CRefXs outData, CRefXd predData)
{
    lossFunValidate_(outData, predData);
    const double falsePos = ((1 - outData).cast<double>() * predData).sum();
    const double falseNeg = (outData.cast<double>() * (1.0 - predData)).sum();
    return { falsePos, falseNeg, falsePos + falseNeg };
}

inline Array3d logLoss(CRefXs outData, CRefXd predData, double gamma = 0.1)
{
    lossFunValidate_(outData, predData);
    if (!(gamma > 0 && gamma <= 1.0))
        throw std::invalid_argument("gamma must lie in the interval (0.0, 1.0].");
    const double falsePos = ((1 - outData).cast<double>() * (1.0 - (1.0 - predData).pow(gamma))).sum() / gamma;
    const double falseNeg = (outData.cast<double>() * (1.0 - predData.pow(gamma))).sum() / gamma;
    return { falsePos, falseNeg, falsePos + falseNeg };
}


inline Array3d auc(CRefXs outData, CRefXd predData)
{
    lossFunValidate_(outData, predData);

    const size_t sampleCount = outData.rows();

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
    ErrorCount(double threshold) : threshold_(threshold)
    {
        if (!(threshold > 0 && threshold < 1.0))
            throw std::invalid_argument("threshold must lie in the interval (0.0, 1.0).");
    }

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


class SenseSpec {
public:
    SenseSpec(double threshold) : threshold_(threshold)
    {
        if (!(threshold > 0 && threshold < 1.0))
            throw std::invalid_argument("threshold must lie in the interval (0.0, 1.0).");
    }

    Array3d operator()(CRefXs outData, CRefXd predData) const
    {
        return senseSpec(outData, predData, threshold_);
    }

    string name() const
    {
        stringstream ss;
        ss << "sens-spec(" << threshold_ << ")";
        return ss.str();
    }

private:
    const double threshold_;
};


class LogLoss {
public:
    LogLoss(double gamma) : gamma_(gamma)
    {
        if (!(gamma > 0 && gamma <= 1.0))
            throw std::invalid_argument("gamma must lie in the interval (0.0, 1.0].");
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

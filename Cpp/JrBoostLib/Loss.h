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

//----------------------------------------------------------------------------------------------------------------------

inline double linLoss(CRefXs outData, CRefXd predData)
{
    lossFunValidate_(outData, predData);
    const double falsePos = ((1 - outData).cast<double>() * predData).sum();
    const double falseNeg = (outData.cast<double>() * (1.0 - predData)).sum();
    return falsePos + falseNeg;
}

inline double linLossWeighted(CRefXs outData, CRefXd predData, CRefXd weights)
{
    lossFunValidate_(outData, predData);
    const double falsePos = (weights * (1 - outData).cast<double>() * predData).sum();
    const double falseNeg = (weights * outData.cast<double>() * (1.0 - predData)).sum();
    return falsePos + falseNeg;
}

//----------------------------------------------------------------------------------------------------------------------

inline double logLoss(CRefXs outData, CRefXd predData, double gamma = 0.001)
{
    lossFunValidate_(outData, predData);
    if (!(gamma > 0 && gamma <= 1.0))
        throw std::invalid_argument("gamma must lie in the interval (0.0, 1.0].");
    const double falsePos = ((1 - outData).cast<double>() * (1.0 - (1.0 - predData).pow(gamma))).sum() / gamma;
    const double falseNeg = (outData.cast<double>() * (1.0 - predData.pow(gamma))).sum() / gamma;
    return falsePos + falseNeg;
}

inline double logLossWeighted(CRefXs outData, CRefXd predData, CRefXd weights, double gamma = 0.001)
{
    lossFunValidate_(outData, predData);
    if (!(gamma > 0 && gamma <= 1.0))
        throw std::invalid_argument("gamma must lie in the interval (0.0, 1.0].");
    const double falsePos = (weights * (1 - outData).cast<double>() * (1.0 - (1.0 - predData).pow(gamma))).sum() / gamma;
    const double falseNeg = (weights * outData.cast<double>() * (1.0 - predData.pow(gamma))).sum() / gamma;
    return falsePos + falseNeg;
}

class LogLoss {
public:
    LogLoss(double gamma) : gamma_(gamma)
    {
        if (!(gamma > 0 && gamma <= 1.0))
            throw std::invalid_argument("gamma must lie in the interval (0.0, 1.0].");
    }

    double operator()(CRefXs outData, CRefXd predData) const
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

class LogLossWeighted {
public:
    LogLossWeighted(double gamma) : gamma_(gamma)
    {
        if (!(gamma > 0 && gamma <= 1.0))
            throw std::invalid_argument("gamma must lie in the interval (0.0, 1.0].");
    }

    double operator()(CRefXs outData, CRefXd predData, CRefXd weights) const
    {
        return logLossWeighted(outData, predData, weights, gamma_);
    }

    string name() const
    {
        stringstream ss;
        ss << "log-loss-w(" << gamma_ << ")";
        return ss.str();
    }

private:
    const double gamma_;
};

//----------------------------------------------------------------------------------------------------------------------

inline double auc(CRefXs outData, CRefXd predData)
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
    return auc;
}

inline double aucWeighted(CRefXs outData, CRefXd predData, CRefXd weights)
{
    lossFunValidate_(outData, predData);

    const size_t sampleCount = outData.rows();

    vector<tuple<size_t, double, double>> tmp(sampleCount);
    for (size_t i = 0; i < sampleCount; ++i)
        tmp[i] = { outData[i], predData[i], weights[i] };
    pdqsort_branchless(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return std::get<1>(x) < std::get<1>(y); });

    double a = 0.0;
    double b = 0.0;
    double totalWeight = 0.0;

    for (auto [y, pred, w] : tmp) {
        //if (y == 0)
        //    a += w;
        //else
        //    b += w * a;
        a += w - y * w;
        b += (y * w) * a;
        totalWeight += w;
    }
    double auc = b / (a * (totalWeight - a));
    return auc;
}

inline double aoc(CRefXs outData, CRefXd predData)
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
        //    b += a;
        //else
        //    a += 1;
        a += y;
        b += (1 - y) * a;
    }
    double auc = static_cast<double>(b) / (static_cast<double>(a) * static_cast<double>(sampleCount - a));
    return auc;
}

inline double aocWeighted(CRefXs outData, CRefXd predData, CRefXd weights)
{
    lossFunValidate_(outData, predData);

    const size_t sampleCount = outData.rows();

    vector<tuple<size_t, double, double>> tmp(sampleCount);
    for (size_t i = 0; i < sampleCount; ++i)
        tmp[i] = { outData[i], predData[i], weights[i] };
    pdqsort_branchless(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return std::get<1>(x) < std::get<1>(y); });

    double a = 0.0;
    double b = 0.0;
    double totalWeight = 0.0;

    for (auto [y, pred, w] : tmp) {
        //if (y == 0)
        //    b += w * a;
        //else
        //    a += w;
        a += y * w;
        b += (w - y * w) * a;
        totalWeight += w;
    }
    double auc = b / (a * (totalWeight - a));
    return auc;
}

inline double negAuc(CRefXs outData, CRefXd predData)
{
    return -auc(outData, predData);
}

inline double negAucWeighted(CRefXs outData, CRefXd predData, CRefXd weights)
{
    return -aucWeighted(outData, predData, weights);
}

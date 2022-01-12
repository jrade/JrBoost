//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "Loss.h"


double linLoss(CRefXu8 outData, CRefXd predData, optional<CRefXd> optWeights)
{
    double loss;
    GUARDED_PROFILE_PUSH(PROFILE::LOSS);
    ITEM_COUNT = outData.rows();

    if (outData.rows() != predData.rows())
        throw std::invalid_argument("True outdata and predicted outdata have different numbers of samples.");
    if ((outData > 1).any())
        throw std::invalid_argument("True outdata has values that are not 0 or 1.");
    if (!predData.isFinite().all())
        throw std::invalid_argument("Predicted outdata has values that are infinity or NaN.");
    if (!(predData >= 0.0f && predData <= 1.0f).all())
        throw std::invalid_argument("Predicted outdata has values that do not lie in the interval [0.0, 1.0].");

    if (optWeights) {
        CRefXd weights = *optWeights;
        if (!weights.isFinite().all())
            throw std::invalid_argument("Weights has values that are infinity or NaN.");
        if (!(weights > 0.0).all())
            throw std::invalid_argument("Weights has non-positive values.");

        const double falsePos = (weights * (1.0 - outData.cast<double>()) * predData).sum();
        const double falseNeg = (weights * outData.cast<double>() * (1.0 - predData)).sum();
        loss = falsePos + falseNeg;
    }
    else {
        const double falsePos = ((1.0 - outData.cast<double>()) * predData).sum();
        const double falseNeg = (outData.cast<double>() * (1.0 - predData)).sum();
        loss = falsePos + falseNeg;
    }

    GUARDED_PROFILE_POP;
    return loss;
}

//----------------------------------------------------------------------------------------------------------------------

double logLoss_(CRefXu8 outData, CRefXd predData, optional<CRefXd> optWeights, double gamma)
{
    double loss;
    GUARDED_PROFILE_PUSH(PROFILE::LOSS);
    ITEM_COUNT = outData.rows();

    if (outData.rows() != predData.rows())
        throw std::invalid_argument("True outdata and predicted outdata have different numbers of samples.");
    if ((outData > 1).any())
        throw std::invalid_argument("True outdata has values that are not 0 or 1.");
    if (!predData.isFinite().all())
        throw std::invalid_argument("Predicted outdata has values that are infinity or NaN.");
    if (!(predData >= 0.0f && predData <= 1.0f).all())
        throw std::invalid_argument("Predicted outdata has values that do not lie in the interval [0.0, 1.0].");
    if (!(gamma > 0 && gamma <= 1.0))   // carefully written to trap NaN
        throw std::invalid_argument("gamma must lie in the interval (0.0, 1.0].");

    if (optWeights) {
        CRefXd weights = *optWeights;
        if (!(weights > 0.0).all())
            throw std::invalid_argument("Weights has non-positive values.");
        if (!(gamma > 0 && gamma <= 1.0))   // carefully written to trap NaN
            throw std::invalid_argument("gamma must lie in the interval (0.0, 1.0].");

        const double falsePos
            = (weights * (1.0 - outData.cast<double>()) * (1.0 - (1.0 - predData).pow(gamma))).sum() / gamma;
        const double falseNeg = (weights * outData.cast<double>() * (1.0 - predData.pow(gamma))).sum() / gamma;
        loss = falsePos + falseNeg;
    }
    else {
        const double falsePos = ((1.0 - outData.cast<double>()) * (1.0 - (1.0 - predData).pow(gamma))).sum() / gamma;
        const double falseNeg = (outData.cast<double>() * (1.0 - predData.pow(gamma))).sum() / gamma;
        loss = falsePos + falseNeg;
    }

    GUARDED_PROFILE_POP;
    return loss;
}

//......................................................................................................................

LogLoss::LogLoss(double gamma) : gamma_(gamma)
{
    if (!(gamma > 0 && gamma <= 1.0))   // carefully written to trap NaN
        throw std::invalid_argument("gamma must lie in the interval (0.0, 1.0].");
}

double LogLoss::operator()(CRefXu8 outData, CRefXd predData, optional<CRefXd> weights) const
{
    return logLoss_(outData, predData, weights, gamma_);
}

string LogLoss::name() const
{
    stringstream ss;
    ss << "log-loss(" << gamma_ << ")";
    return ss.str();
}

//----------------------------------------------------------------------------------------------------------------------

double aucNoWeights_(CRefXu8 outData, CRefXd predData);
double aucWeights_(CRefXu8 outData, CRefXd predData, CRefXd weights);

double auc(CRefXu8 outData, CRefXd predData, optional<CRefXd> weights)
{
    double gain;
    GUARDED_PROFILE_PUSH(PROFILE::LOSS);
    ITEM_COUNT = outData.rows();

    if (outData.rows() != predData.rows())
        throw std::invalid_argument("True outdata and predicted outdata have different numbers of samples.");
    if ((outData > 1).any())
        throw std::invalid_argument("True outdata has values that are not 0 or 1.");
    if (!predData.isFinite().all())
        throw std::invalid_argument("Predicted outdata has values that are infinity or NaN.");

    if (weights)
        gain = aucWeights_(outData, predData, *weights);
    else
        gain = aucNoWeights_(outData, predData);

    GUARDED_PROFILE_POP;
    return gain;
}

double aucNoWeights_(CRefXu8 outData, CRefXd predData)
{
    const size_t sampleCount = outData.rows();
    vector<pair<uint8_t, double>> tmp(sampleCount);
    for (size_t i = 0; i != sampleCount; ++i)
        tmp[i] = {outData[i], predData[i]};
    pdqsort_branchless(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return x.second < y.second; });

    size_t a = 0;
    size_t b = 0;
    for (auto [y, pred] : tmp) {
        // if (y == 0)
        //    a += 1;
        // else
        //    b += a;
        a += 1 - y;
        b += y * a;
    }

    return static_cast<double>(b) / (static_cast<double>(a) * static_cast<double>(sampleCount - a));
}

double aucWeights_(CRefXu8 outData, CRefXd predData, CRefXd weights)
{
    if (!weights.isFinite().all())
        throw std::invalid_argument("Weights has values that are infinity or NaN.");
    if (!(weights > 0.0).all())
        throw std::invalid_argument("Weights has non-positive values.");

    const size_t sampleCount = outData.rows();
    vector<tuple<uint8_t, double, double>> tmp(sampleCount);
    for (size_t i = 0; i != sampleCount; ++i)
        tmp[i] = {outData[i], predData[i], weights[i]};
    pdqsort_branchless(
        begin(tmp), end(tmp), [](const auto& x, const auto& y) { return std::get<1>(x) < std::get<1>(y); });

    double a = 0.0;
    double b = 0.0;
    double totalWeight = 0.0;

    for (auto [y, pred, w] : tmp) {
        // if (y == 0)
        //    a += w;
        // else
        //    b += w * a;
        a += w - y * w;
        b += (y * w) * a;
        totalWeight += w;
    }

    return b / (a * (totalWeight - a));
}

//......................................................................................................................

double aocNoWeights_(CRefXu8 outData, CRefXd predData);
double aocWeights_(CRefXu8 outData, CRefXd predData, CRefXd weights);

double aoc(CRefXu8 outData, CRefXd predData, optional<CRefXd> weights)
{
    double loss;
    GUARDED_PROFILE_PUSH(PROFILE::LOSS);

    if (outData.rows() != predData.rows())
        throw std::invalid_argument("True outdata and predicted outdata have different numbers of samples.");
    if ((outData > 1).any())
        throw std::invalid_argument("True outdata has values that are not 0 or 1.");
    if (!predData.isFinite().all())
        throw std::invalid_argument("Predicted outdata has values that are infinity or NaN.");

    if (weights)
        loss = aocWeights_(outData, predData, *weights);
    else
        loss = aocNoWeights_(outData, predData);

    GUARDED_PROFILE_POP;
    return loss;
}

double aocNoWeights_(CRefXu8 outData, CRefXd predData)
{
    const size_t sampleCount = outData.rows();
    vector<pair<uint8_t, double>> tmp(sampleCount);
    for (size_t i = 0; i != sampleCount; ++i)
        tmp[i] = {outData[i], predData[i]};
    pdqsort_branchless(begin(tmp), end(tmp), [](const auto& x, const auto& y) { return x.second < y.second; });

    size_t a = 0;
    size_t b = 0;
    for (auto [y, pred] : tmp) {
        // if (y == 0)
        //    b += a;
        // else
        //    a += 1;
        a += y;
        b += (1 - y) * a;
    }

    return static_cast<double>(b) / (static_cast<double>(a) * static_cast<double>(sampleCount - a));
}

double aocWeights_(CRefXu8 outData, CRefXd predData, CRefXd weights)
{
    if (!weights.isFinite().all())
        throw std::invalid_argument("Weights has values that are infinity or NaN.");
    if (!(weights > 0.0).all())
        throw std::invalid_argument("Weights has non-positive values.");

    const size_t sampleCount = outData.rows();
    vector<tuple<uint8_t, double, double>> tmp(sampleCount);
    for (size_t i = 0; i != sampleCount; ++i)
        tmp[i] = {outData[i], predData[i], weights[i]};
    pdqsort_branchless(
        begin(tmp), end(tmp), [](const auto& x, const auto& y) { return std::get<1>(x) < std::get<1>(y); });

    double a = 0.0;
    double b = 0.0;
    double totalWeight = 0.0;

    for (auto [y, pred, w] : tmp) {
        // if (y == 0)
        //    b += w * a;
        // else
        //    a += w;
        a += y * w;
        b += (w - y * w) * a;
        totalWeight += w;
    }

    return b / (a * (totalWeight - a));
}

//......................................................................................................................

double negAuc(CRefXu8 outData, CRefXd predData, optional<CRefXd> weights) { return -auc(outData, predData, weights); }

#pragma once


inline tuple<double, double, double> errorCount(CRefXs outData, CRefXd predData, double p)
{
#ifdef USE_LOR
    double cutoff = std::log(p) - std::log(1.0 - p);
#else
    double cutoff = p;
#endif
    double falsePos = static_cast<double>(((1 - outData) * (predData >= cutoff).cast<size_t>()).sum());
    double falseNeg = static_cast<double>((outData * (predData < cutoff).cast<size_t>()).sum());
    return tuple{ falsePos, falseNeg, falsePos + falseNeg };
}


inline tuple<double, double, double> linLoss(CRefXs outData, CRefXd predData)
{
#ifdef USE_LOR
    double falsePos = ((1 - outData).cast<double>() / (1.0 + (-predData).exp())).sum();
    double falseNeg = (outData.cast<double>() / (1.0 + predData.exp())).sum();
#else
    double falsePos = ((1 - outData).cast<double>() * predData).sum();
    double falseNeg = (outData.cast<double>() * (1.0 - predData)).sum();
#endif
    return std::make_tuple(falsePos, falseNeg, falsePos + falseNeg);
}


inline tuple<double, double, double> logLoss(CRefXs outData, CRefXd predData)
{
#ifdef USE_LOR

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

#else
    constexpr double epsilon = std::numeric_limits<double>::min();
    double falsePos = -((1 - outData).cast<double>() * (1.0 - predData + epsilon).log()).sum();
    double falseNeg = -(outData.cast<double>() * (predData + epsilon).log()).sum();
#endif
    return std::make_tuple(falsePos, falseNeg, falsePos + falseNeg);
}


inline tuple<double, double, double> alphaLoss(CRefXs outData, CRefXd predData, double alpha)
{
#ifdef USE_LOR

    double falsePos = ((1 - outData).cast<double>()
        * (1.0 - (1.0 / (1.0 + predData.exp())).pow(alpha))
    ).sum() / alpha;

    double falseNeg = (outData.cast<double>()
        * (1.0 - (1.0 / (1.0 + (-predData).exp())).pow(alpha))
    ).sum() / alpha;

#else
    double falsePos = ((1 - outData).cast<double>() * (1.0 - (1.0 - predData).pow(alpha))).sum() / alpha;
    double falseNeg = (outData.cast<double>() * (1.0 - predData.pow(alpha))).sum() / alpha;
#endif
    return std::make_tuple(falsePos, falseNeg, falsePos + falseNeg);
}


inline tuple<double, double, double> sqrtLoss(CRefXs outData, CRefXd predData)
{
    return alphaLoss(outData, predData, 0.5);
}


//----------------------------------------------------------------------------------------------------------------------


class ErrorCount {
public:
    ErrorCount(double p) : p_(p) {}

    tuple<double, double, double> operator()(CRefXs outData, CRefXd predData) const
    {
        return errorCount(outData, predData, p_);
    }

    string name() const
    {
        stringstream ss;
        ss << "err-count(" << p_ << ")";
        return ss.str();
    }

private:
    const double p_;
};


class AlphaLoss {
public:
    AlphaLoss(double alpha) : alpha_(alpha) {}

    tuple<double, double, double> operator()(CRefXs outData, CRefXd predData) const
    {
        return alphaLoss(outData, predData, alpha_);
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

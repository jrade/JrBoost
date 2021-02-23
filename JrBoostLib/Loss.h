#pragma once


inline tuple<double, double, double> linLoss(CRefXs outData, CRefXd predData)
{
    double falsePos = ((1 - outData).cast<double>() / (1.0 + (-predData).exp())).sum();
    double falseNeg = (outData.cast<double>() / (1.0 + predData.exp())).sum();
    return std::make_tuple(falsePos, falseNeg, falsePos + falseNeg);
}


inline tuple<double, double, double> logLoss(CRefXs outData, CRefXd predData)
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

    return std::make_tuple(falsePos, falseNeg, falsePos + falseNeg);
}


inline tuple<double, double, double> alphaLoss(CRefXs outData, CRefXd predData, double alpha)
{
    double falsePos = ((1 - outData).cast<double>()
        * (1.0 - (1.0 / (1.0 + predData.exp())).pow(alpha))
    ).sum() / alpha;

    double falseNeg = (outData.cast<double>()
        * (1.0 - (1.0 / (1.0 + (-predData).exp())).pow(alpha))
    ).sum() / alpha;

    return std::make_tuple(falsePos, falseNeg, falsePos + falseNeg);
}


inline tuple<double, double, double> sqrtLoss(CRefXs outData, CRefXd predData)
{
    return alphaLoss(outData, predData, 0.5);
}


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

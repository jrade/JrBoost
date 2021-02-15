#pragma once


//inline double linLoss(CRefXs outData, CRefXd predData)
//{
//    return (
//        outData.cast<double>() / (1.0 + predData.exp())
//        + (1 - outData).cast<double>() / (1 + (-predData).exp())
//    ).sum();
//}


inline tuple<double, double, double> linLoss(CRefXs outData, CRefXd predData)
{
    double falsePos = ((1 - outData).cast<double>() / (1 + (-predData).exp())).sum();
    double falseNeg = (outData.cast<double>() / (1.0 + predData.exp())).sum();
    return std::make_tuple(falsePos, falseNeg, falsePos + falseNeg);
}


template<typename T, typename U, typename F>
inline void sortedIndices(T p0, T p1, U q0, F f)
{
    using R = decltype(f(*p0));

    vector<pair<size_t, R>> tmp(p1 - p0);
    auto r0 = begin(tmp);
    auto r1 = end(tmp);

    size_t i = 0;
    auto p = p0;
    auto r = r0;
    while (p != p1)
        *(r++) = std::make_pair(i++, f(*(p++)));
    std::sort(
        r0, 
        r1,
        [](const auto& x, const auto& y) { return x.second < y.second; }
    );

    r = r0;
    auto q = q0;
    while (r != r1)
        *(q++) = (r++)->first;
}


inline ArrayXs tStatisticRank(
    Eigen::Ref<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> inData,
    CRefXs outData,
    optional<CRefXs> optSamples
)
{
    PROFILE::PUSH(PROFILE::T_RANK);

    size_t variableCount = inData.cols();    
    ASSERT(outData.rows() == inData.rows());

    ArrayXs samples;
    size_t sampleCount;
    if (optSamples) {
        samples = *optSamples;
        sampleCount = samples.size();
    }
    else {
        sampleCount = inData.rows();
        samples.resize(sampleCount);
        for (size_t i = 0; i < sampleCount; ++i)
            samples(i) = i;
    }

    size_t n[2] = { 0, 0 };
    ArrayXf mean[2];
    mean[0] = ArrayXf::Constant(variableCount, 0.0f);
    mean[1] = ArrayXf::Constant(variableCount, 0.0f);
    for (int i = 0; i < sampleCount; ++i) {
        size_t j = samples[i];
        size_t k = outData(j);
        ++n[k];
        mean[k] += inData.row(j);
    }
    for (int k = 0; k < 2; ++k)
        mean[k] /= static_cast<float>(n[k]);

    ArrayXf SS[2];
    SS[0] = ArrayXf::Constant(variableCount, 0.0f);
    SS[1] = ArrayXf::Constant(variableCount, 0.0f);
    for (int i = 0; i < sampleCount; ++i) {
        size_t j = samples[i];
        size_t k = outData(j);
        SS[k] += (inData.row(j) - mean[k].transpose()).square();
    }

    ArrayXf t2 = (mean[1] - mean[0]).square() / (SS[0] + SS[1] + numeric_limits<float>::min());

    ArrayXs ind(variableCount);
    sortedIndices(
        &t2(0),
        &t2(0) + variableCount,
        &ind(0),
        [](auto x) { return -x; }
    );

    PROFILE::POP(sampleCount * variableCount);

    return ind;
}

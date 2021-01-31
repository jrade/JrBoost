#pragma once


inline double linLoss(CRefXs outData, CRefXd predData)
{
    return (
        outData.cast<double>() / (1.0 + predData.exp())
        + (1 - outData).cast<double>() / (1 + (-predData).exp())
    ).sum();
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
    CRefXs outData
)
{
    CLOCK::PUSH(CLOCK::T_RANK);
        
    size_t variableCount = inData.cols();
    size_t sampleCount = inData.rows();

    size_t n0 = 0;
    size_t n1 = 0;
    ArrayXf mean0 = ArrayXf::Constant(variableCount, 0.0f);
    ArrayXf mean1 = ArrayXf::Constant(variableCount, 0.0f);
    for (int i = 0; i < sampleCount; ++i) {
        size_t k = outData(i);
        if (k == 0) {
            ++n0;
            mean0 += inData.row(i);
        }
        else if (k == 1) {
            ++n1;
            mean1 += inData.row(i);
        }
    }
    mean0 /= static_cast<float>(n0);
    mean1 /= static_cast<float>(n1);

    ArrayXf SS0 = ArrayXf::Constant(variableCount, 0.0f);
    ArrayXf SS1 = ArrayXf::Constant(variableCount, 0.0f);
    for (int i = 0; i < sampleCount; ++i) {
        size_t k = outData(i);
        if (k == 0)
            SS0 += (inData.row(i) - mean0.transpose()).square();
        else if (k == 1)
            SS1 += (inData.row(i) - mean1.transpose()).square();
    }

    ArrayXf t2 = (mean1 - mean0).square() / (SS0 + SS1 + numeric_limits<float>::min());

    ArrayXs ind(variableCount);
    sortedIndices(
        &t2(0),
        &t2(0) + variableCount,
        &ind(0),
        [](auto x) { return -x; }
    );

    CLOCK::POP((n0 + n1) * variableCount);

    return ind;
}

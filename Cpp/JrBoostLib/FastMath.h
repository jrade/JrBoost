#pragma once

constexpr double c1__ = (1ll << 52) / 0.6931471805599453;
constexpr double c2__ = (1ll << 52) * (1023 - 0.04367744890362246);
constexpr double c3__ = (1ll << 52);
constexpr double c4__ = (1ll << 52) * 2047;

//----------------------------------------------------------------------------------------------------------------------

// fast approximate exponential with relative error < 3%

inline double fastExp(double x)
{
    x = c1__ * x + c2__;
    x = (x < c3__) ? 0 : x;
    size_t n = static_cast<size_t>(x);
    memcpy(&x, &n, 8);
    return x;
}

// fast approximate pow

inline double fastPow(double x, double a)
{
    size_t n;
    memcpy(&n, &x, 8);
    x = static_cast<double>(n);
    x = a * x + (1 - a) * c2__;
    n = static_cast<size_t>(x);
    memcpy(&x, &n, 8);
    return x;
}

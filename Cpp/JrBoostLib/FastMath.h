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

inline vcl::Vec4d fastExp(vcl::Vec4d x)
{
    x = c1__ * x + c2__;
    x = vcl::select(x < c3__, 0, x);
    //x = vcl::select(x > c4, c4, x);
    vcl::Vec4q n = vcl::roundi(x);
    x = vcl::reinterpret_d(n);
    return x;
}

inline vcl::Vec8d fastExp(vcl::Vec8d x)
{
    x = c1__ * x + c2__;
    x = vcl::select(x < c3__, 0, x);
    //x = vcl::select(x > c4, c4, x);
    vcl::Vec8q n = vcl::roundi(x);
    x = vcl::reinterpret_d(n);
    return x;
}

inline ArrayXd fastExp(ArrayXd&& x)
{
    x = c1__ * x + c2__;
    x = (x < c3__).select(0, x);
    reinterpret_cast<ArrayXs&>(x) = x.cast<size_t>();
    return x;
}

inline ArrayXd fastExp(const ArrayXd& x)
{
    ArrayXd y = c1__ * x + c2__;
    y = (y < c3__).select(0, y);
    reinterpret_cast<ArrayXs&>(y) = y.cast<size_t>();
    return y;
}

//----------------------------------------------------------------------------------------------------------------------

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

inline vcl::Vec4d fastPow(vcl::Vec4d x, double a)
{
    vcl::Vec4q n = vcl::reinterpret_i(x);
    x = vcl::to_double(n);
    x = a * x + (1 - a) * c2__;
    n = vcl::roundi(x);
    x = vcl::reinterpret_d(n);
    return x;
}

inline vcl::Vec8d fastPow(vcl::Vec8d x, double a)
{
    vcl::Vec8q n = vcl::reinterpret_i(x);
    x = vcl::to_double(n);
    x = a * x + (1 - a) * c2__;
    n = vcl::roundi(x);
    x = vcl::reinterpret_d(n);
    return x;
}

inline ArrayXd fastPow(ArrayXd&& x, double a)
{
    x = (reinterpret_cast<ArrayXs&>(x)).cast<double>();
    x = a * x + (1 - a) * c2__;
    reinterpret_cast<ArrayXs&>(x) = x.cast<size_t>();
    return x;
}

inline ArrayXd fastPow(const ArrayXd& x, double a)
{
    ArrayXd y = (reinterpret_cast<const ArrayXs&>(x)).cast<double>();
    y = a * y + (1 - a) * c2__;
    reinterpret_cast<ArrayXs&>(y) = y.cast<size_t>();
    return y;
}

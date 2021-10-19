//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


inline double fastExp(double x)
{
    constexpr double a = (1ll << 52) / 0.6931471805599453;
    constexpr double b = (1ll << 52) * (1023 - 0.04367744890362246);
    x = a * x + b;

    // if underflow, return 0
    x = std::max(x, 0.0);

    // if overflow, return positive infinity
    constexpr double c = (1ll << 52) * 2047;
    x = std::min(x, c);

    int64_t n = static_cast<int64_t>(x);

    memcpy(&x, &n, 8);
    // using memcpy is standard compliant and as fast as UB hacks with reinterpret_cast or union

    return x;
}

//----------------------------------------------------------------------------------------------------------------------

#if defined(__AVX512F__) && defined(__AVX512DQ__)

// WARNING: NOT TESTED!

inline __m256d fastExp(__m256d x4)
{
    constexpr double a = (1ll << 52) / 0.6931471805599453;
    constexpr double b = (1ll << 52) * (1023 - 0.04367744890362246);
    constexpr __m256d a4 = { a, a, a, a };
    constexpr __m256d b4 = { b, b, b, b };
    x4 = _mm256_fmadd_pd(x4, a4, b4);             // x = x * a + b

    // if underflow, return 0
    constexpr __m256d zero4 = { 0.0, 0.0, 0.0, 0.0 };
    x4 = _mm256_max_pd(x4, zero4);                // x = std::max(x, 0.0)

    // if overflow, return positive infinity
    constexpr double c = (1ll << 52) * 2047;
    constexpr __m256d c4 = { c, c, c, c };
    x4 = _mm256_min_pd(x4, c4);                   // x = std::min(x, c)

    __m256i n4 = _mm256_cvtpd_epi64(x4);          // int64_t n = static_cast<int64_t>(x)

    x4 = _mm256_castsi256_pd(n4);                 // memcpy(&x, &n, 8)

    return x4;
}

#elif defined(__AVX2__)

inline __m256d fastExp(__m256d x4)
{
    // AVX2 has no double to int64_t conversion, so a workaround is needed

    constexpr double a = (1ll << 20) / 0.6931471805599453;
    constexpr double b = (1ll << 20) * (1023 - 0.04367744890362246) + 0.5;
    constexpr __m256d a4 = { a, a, a, a };
    constexpr __m256d b4 = { b, b, b, b };
    x4 = _mm256_fmadd_pd(x4, a4, b4);             // x = x * a + b

    // if underflow, return 0
    constexpr __m256d zero4 = { 0.0, 0.0, 0.0, 0.0 };
    x4 = _mm256_max_pd(x4, zero4);                // x = std::max(x, 0.0)

    // if overflow, return positive infinity
    constexpr double c = (1ll << 20) * 2047;
    constexpr __m256d c4 = { c, c, c, c };
    x4 = _mm256_min_pd(x4, c4);                   // x = std::min(x, c)

    __m128i m4 = _mm256_cvtpd_epi32(x4);          // int32_t m = static_cast<int32_t>(x)
    __m256i n4 = _mm256_cvtepi32_epi64(m4);       // int64_t n = static_cast<int64_t>(m) 
    n4 = _mm256_slli_epi64(n4, 32);               // n = n << 32

    x4 = _mm256_castsi256_pd(n4);                // memcpy(&x, &n, 8)

    return x4;
}

#endif

//----------------------------------------------------------------------------------------------------------------------

#if defined(__AVX512F__) && defined(__AVX512DQ__)

// WARNING: NOT TESTED!

inline __m512d fastExp(__m512d x8)
{
    constexpr double a = (1ll << 52) / 0.6931471805599453;
    constexpr double b = (1ll << 52) * (1023 - 0.04367744890362246);
    constexpr __m512d a8 = { a, a, a, a, a, a, a, a };
    constexpr __m512d b8 = { b, b, b, b, b, b, b, b };
    x8 = _mm512_fmadd_pd(x8, a8, b8);             // x = x * a + b

    // if underflow, return 0
    constexpr __m512d zero8 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    x8 = _mm512_max_pd(x8, zero8);                // x = std::max(x, 0.0)

    // if overflow, return positive infinity
    constexpr double c = (1ll << 52) * 2047;
    constexpr __m512d c8 = { c, c, c, c, c, c, c, c };
    x8 = _mm512_min_pd(x8, c8);                   // x = std::min(x, c)

    __m512i n8 = _mm512_cvtpd_epi64(x8);          // int64_t n = static_cast<int64_t>(x)

    x8 = _mm512_castsi512_pd(n8);                 // memcpy(&x, &n, 8)

    return x8;
}

#endif

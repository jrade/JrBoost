//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt 
//  or copy at https://opensource.org/licenses/MIT)

// Very fast approximate implementation of the exponential function (exp) with
//     relative error < 3%
//     underflow and overflow checks
//     single and double precision versions
//     SIMD versions (using Intel Intrinsics)

// I use the method described in the paper
//     Nicol N. Schraudolph
//     A Fast, Compact Approximation of the Exponential Function
//     Neural Computation 11, 853–862 (1999)
// available at https://www.schraudolph.org/pubs/Schraudolph99.pdf
// The method is based on clever manipulation of the bits in the IEEE754
// binary representation of floating point numbers.
//  
// I have added some features that are not in the paper, such as
// underflow and overflow checks, single precision versions and SIMD versions.
// The underflow and  overflow checks use the bit manipulation techniques
// described in the paper; the constant 2047LL << 52 in the overflow check is
// the 64-bit integer with the same binary representation as double precision
// positive infinity.

//------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

//------------------------------------------------------------------------------

// The double precision versions of fastExp(x) behave as follows:
//       -inf <= x < -709.06:   return value = 0
//    -709.06 <  x < -708.37:   absolute error < 1e-307
//    -708.37 <  x <  709.81:   relative error < 0.0299
//     709.81 <  x <=    inf:   return value = inf
//               x =     nan:   unspecified return value

inline double fastExp(double x)
{
    const double a = (1LL << 52) / 0.6931471805599453;
    const double b = (1LL << 52) * (1023 - 0.0436774489036);
    x = a * x + b;

    // if overflow, return positive infinity
    const double c = 2047LL << 52;
    x = std::min(x, c);

    // if underflow, return 0
    x = std::max(x, 0.0);

    int64_t n = static_cast<int64_t>(x);

    std::memcpy(&x, &n, 8);
    // using memcpy is standard compliant and as fast as UB hacks
    // with reinterpret_cast or union

    return x;
}

inline __m512d fastExp(__m512d x8)          // requires AVX512F + AVX512DQ
{
    const double a = (1LL << 52) / 0.6931471805599453;
    const double b = (1LL << 52) * (1023 - 0.0436774489036);
    x8 = _mm512_fmadd_pd(x8, _mm512_set1_pd(a), _mm512_set1_pd(b));

    // if overflow, return positive infinity
    const double c = 2047LL << 52;
    x8 = _mm512_min_pd(x8, _mm512_set1_pd(c));

    // if underflow, return 0
    x8 = _mm512_max_pd(x8, _mm512_setzero_pd());

    __m512i n8 = _mm512_cvtpd_epi64(x8);

    x8 = _mm512_castsi512_pd(n8);

    return x8;
}

inline __m256d fastExp(__m256d x4)          // requires AVX2
{
    // AVX2 does not support conversion from double to int64_t.
    // This implementation avoids that conversion.

    const double a = (1LL << 20) / 0.6931471805599453;
    const double b = (1LL << 20) * (1023 - 0.0436774489036) + 0.5;
    x4 = _mm256_fmadd_pd(x4, _mm256_set1_pd(a), _mm256_set1_pd(b));

    // if overflow, return positive infinity
    const double c = 2047LL << 20;
    x4 = _mm256_min_pd(x4, _mm256_set1_pd(c));

    // if underflow, return 0
    x4 = _mm256_max_pd(x4, _mm256_setzero_pd());

    __m128i m4 = _mm256_cvtpd_epi32(x4);
    __m256i n4 = _mm256_cvtepi32_epi64(m4);
    n4 = _mm256_slli_epi64(n4, 32);

    x4 = _mm256_castsi256_pd(n4);

    return x4;

/*
    A scalar version of this code would be
        x = a * x + b;
        x = std::min(x, c);
        x = std::max(x, 0.0);
        int32_t m = static_cast<int32_t>(x);
        int64_t n = static_cast<int64_t>(m);
        n <<= 32;
        std::memcpy(&x, &n, 8);
        return x;
*/
}

//------------------------------------------------------------------------------

// The single precision versions of fastExp(x) behave as follows:
//      -inf <= x < -88.00:   return value = 0
//    -88.00 <  x < -87.30:   absolute error < 1e-38
//    -87.30 <  x <  88.72:   relative error < 0.0299
//     88.72 <  x <=   inf:   return value = inf
//              x =    nan:   unspecified return value

inline float fastExp(float x)
{
    constexpr float a = (1 << 23) / 0.6931472f;
    constexpr float b = (1 << 23) * (127 - 0.04368f) + 0.5f;
    x = a * x + b;

    // if overflow, return positive infinity
    const float c = 255 << 23;
    x = std::min(x, c);

    // if underflow, return 0
    x = std::max(x, 0.0f);

    uint32_t n = static_cast<uint32_t>(x);

    memcpy(&x, &n, 4);
    // using memcpy is standard compliant and as fast as UB hacks
    // with reinterpret_cast or union

    return x;
}

inline __m512 fastExp(__m512 x16)           // requires AVX512F + AVX512DQ
{
    constexpr float a = (1 << 23) / 0.6931472f;
    constexpr float b = (1 << 23) * (127 - 0.04368f) + 0.5f;
    x16 = _mm512_fmadd_ps(x16, _mm512_set1_ps(a), _mm512_set1_ps(b));

    // if overflow, return positive infinity
    const float c = 255 << 23;
    x16 = _mm512_min_ps(x16, _mm512_set1_ps(c));

    // if underflow, return 0
    x16 = _mm512_max_ps(x16, _mm512_setzero_ps());

    __m512i n16 = _mm512_cvtps_epi32(x16);

    x16 = _mm512_castsi512_ps(n16);

    return x16;
}

inline __m256 fastExp(__m256 x8)            // requires AVX2
{
    constexpr float a = (1 << 23) / 0.6931472f;
    constexpr float b = (1 << 23) * (127 - 0.04368f) + 0.5f;
    x8 = _mm256_fmadd_ps(x8, _mm256_set1_ps(a), _mm256_set1_ps(b));

    // if overflow, return positive infinity
    const float c = 255 << 23;
    x8 = _mm256_min_ps(x8, _mm256_set1_ps(c));

    // if underflow, return 0
    x8 = _mm256_max_ps(x8, _mm256_setzero_ps());

    __m256i n8 = _mm256_cvtps_epi32(x8);

    x8 = _mm256_castsi256_ps(n8);

    return x8;
}

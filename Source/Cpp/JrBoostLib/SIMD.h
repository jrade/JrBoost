//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

//----------------------------------------------------------------------------------------------------------------------

// These functions require AVX2

template<typename T>
__m256i mm256_set1(T a);

template<>
inline __m256i mm256_set1<int8_t>(int8_t a)
{
    return _mm256_set1_epi8(a);   // AVX
}

template<>
inline __m256i mm256_set1<int16_t>(int16_t a)
{
    return _mm256_set1_epi16(a);   // AVX
}

template<>
inline __m256i mm256_set1<int32_t>(int32_t a)
{
    return _mm256_set1_epi32(a);   // AVX
}

//......................................................................................................................

template<typename T>
__m256i mm256_cmpgt(__m256i a, __m256i b);

template<>
inline __m256i mm256_cmpgt<int8_t>(__m256i a, __m256i b)
{
    return _mm256_cmpgt_epi8(a, b);   // AVX2
}

template<>
inline __m256i mm256_cmpgt<int16_t>(__m256i a, __m256i b)
{
    return _mm256_cmpgt_epi16(a, b);   // AVX2
}

template<>
inline __m256i mm256_cmpgt<int32_t>(__m256i a, __m256i b)
{
    return _mm256_cmpgt_epi32(a, b);   // AVX2
}

//......................................................................................................................

template<typename T>
__m256i mm256_sub(__m256i a, __m256i b);

template<>
inline __m256i mm256_sub<int8_t>(__m256i a, __m256i b)
{
    return _mm256_sub_epi8(a, b);   // AVX2
}

template<>
inline __m256i mm256_sub<int16_t>(__m256i a, __m256i b)
{
    return _mm256_sub_epi16(a, b);   // AVX2
}

template<>
inline __m256i mm256_sub<int32_t>(__m256i a, __m256i b)
{
    return _mm256_sub_epi32(a, b);   // AVX2
}

//----------------------------------------------------------------------------------------------------------------------

// These functions require AVX-512F + AVX-512BW

template<typename T>
struct mmask;

template<>
struct mmask<int8_t> {
    using type = __mmask64;
};

template<>
struct mmask<int16_t> {
    using type = __mmask32;
};

template<>
struct mmask<int32_t> {
    using type = __mmask16;
};

//......................................................................................................................

template<typename T>
__m512i mm512_set1(T a);

template<>
inline __m512i mm512_set1<int8_t>(int8_t a)
{
    return _mm512_set1_epi8(a);   // AVX-512F
}

template<>
inline __m512i mm512_set1<int16_t>(int16_t a)
{
    return _mm512_set1_epi16(a);   // AVX-512F
}

template<>
inline __m512i mm512_set1<int32_t>(int32_t a)
{
    return _mm512_set1_epi32(a);   // AVX-512F
}

//......................................................................................................................

template<typename T>
typename mmask<T>::type mm512_cmpgt_mask(__m512i a, __m512i b);

template<>
typename mmask<int8_t>::type mm512_cmpgt_mask<int8_t>(__m512i a, __m512i b)
{
    return _mm512_cmpgt_epi8_mask(a, b);   // AVX-512BW
}

template<>
typename mmask<int16_t>::type mm512_cmpgt_mask<int16_t>(__m512i a, __m512i b)
{
    return _mm512_cmpgt_epi16_mask(a, b);   // AVX-512BW
}

template<>
typename mmask<int32_t>::type mm512_cmpgt_mask<int32_t>(__m512i a, __m512i b)
{
    return _mm512_cmpgt_epi32_mask(a, b);   // AVX-512F
}

//......................................................................................................................

template<typename T>
__m512i mm512_mask_add(__m512i src, typename mmask<T>::type k, __m512i a, __m512i b);

template<>
__m512i mm512_mask_add<int8_t>(__m512i src, typename mmask<int8_t>::type k, __m512i a, __m512i b)
{
    return _mm512_mask_add_epi8(src, k, a, b);   // AVX-512BW
}

template<>
__m512i mm512_mask_add<int16_t>(__m512i src, typename mmask<int16_t>::type k, __m512i a, __m512i b)
{
    return _mm512_mask_add_epi16(src, k, a, b);   // AVX-512BW
}

template<>
__m512i mm512_mask_add<int32_t>(__m512i src, typename mmask<int32_t>::type k, __m512i a, __m512i b)
{
    return _mm512_mask_add_epi32(src, k, a, b);   // AVX-512F
}

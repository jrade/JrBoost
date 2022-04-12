//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "TopScoringPairs.h"

#include "SIMD.h"

//----------------------------------------------------------------------------------------------------------------------

template<typename Int>
tuple<ArrayXs, ArrayXs, ArrayXd>
topScoringPairsImpl_(CRefXXfr inData, CRefXu8 outData, size_t pairCount, optional<CRefXs> samples);

void validateData_(CRefXXfr inData, CRefXu8 outData, size_t pairCount);

template<typename Int>
pair<ArrayXXr<Int>, ArrayXXr<Int>> preprocessData_(CRefXXfr inData, CRefXu8 outData, optional<CRefXs> samples);

template<typename Int>
inline void rankifyRow_(const float* inData, Int* rankData, pair<float, size_t>* tmp, size_t variableCount);

template<typename Int>
inline void processBlock_(const Int* __restrict p1, const Int* __restrict p2, std::make_unsigned_t<Int>* __restrict q);

//......................................................................................................................

// Data is procesed in blocks of blockSize x blockSize elements of type Int and UInt.
// In the main loop three blocks are processed at a time; two are read and one is written.
// It seems that the best performance is achieved when these blocks fit in the L2 cache
// together with other data and code.
// Maybe the optimal block size (in bytes) is around 1/6 of the L2 cache size.
// Then the block size should roughly satisfy
//     sizeof(Int) * blockSize * blockSize = L2CacheBytesPerCore / 6.
//
// Also if the code is vectorized, then blockSize should be a multiple of the used SIMD register size.
// For simplicity we make it a multiple of the cache line size, which is a multiple of any SIMD register size.

const size_t L2CacheBytesPerCore = 1 << 18;   // 256K, typical value?

inline constexpr size_t roundDownToMultiple_(size_t a, size_t b) { return (a / b) * b; }

template<typename Int>
const size_t blockSize = roundDownToMultiple_(
    ::integerSqrt(::L2CacheBytesPerCore / 6 / sizeof(Int)), std::hardware_destructive_interference_size / sizeof(Int));

//......................................................................................................................

tuple<ArrayXs, ArrayXs, ArrayXd>
topScoringPairs(CRefXXfr inData, CRefXu8 outData, size_t pairCount, optional<CRefXs> samples)
{
    // We select the integer type for the ranks based on the variable count.
    // A more precise way would be to use the maximum of the actual rank count for each variable.
    // One could even consider using different integer types for different variables based on rank count.

    const size_t variableCount = static_cast<size_t>(inData.cols());
    if (variableCount <= 0x100)
        return topScoringPairsImpl_<int8_t>(inData, outData, pairCount, samples);
    else if (variableCount <= 0x10000)
        return topScoringPairsImpl_<int16_t>(inData, outData, pairCount, samples);
    else
        return topScoringPairsImpl_<int32_t>(inData, outData, pairCount, samples);

    // Since the elements of inData are of type float, the number of ranks can never exceed 0x100000000
    // and int32_t will always suffice no matter how many variables.
}


template<typename Int>
tuple<ArrayXs, ArrayXs, ArrayXd>
topScoringPairsImpl_(CRefXXfr inData, CRefXu8 outData, size_t pairCount, optional<CRefXs> samples)
{
    using UInt = std::make_unsigned_t<Int>;

    size_t ITEM_COUNT;
    ScopedProfiler sp(PROFILE::ZERO, &ITEM_COUNT);
    PROFILE::SWITCH(PROFILE::TSP, 0);

    validateData_(inData, outData, pairCount);

    ArrayXXr<Int> zeroInData;
    ArrayXXr<Int> oneInData;
    std::tie(zeroInData, oneInData) = preprocessData_<Int>(inData, outData, samples);

    // preprocessData_() does the following:
    //    discard unused samples
    //    split the data into one array with samples with outData = 0 and and one with outData = 1
    //    replace the values by ranks (using a signed integer type Int as this work better
    //        than an unsigned integer type with the Intel SIMD instruction sets)
    //    pad the rows to a multiple of blockSize<Int>

    const size_t zeroSampleCount = static_cast<size_t>(zeroInData.rows());
    const size_t oneSampleCount = static_cast<size_t>(oneInData.rows());
    const size_t sampleCount = zeroSampleCount + oneSampleCount;

    const size_t variableCount = static_cast<size_t>(inData.cols());
    const size_t blockCount = ::divideRoundUp(variableCount, blockSize<Int>);

    const size_t upperTriangleBlockCount = blockCount * (blockCount + 1) / 2;
    ITEM_COUNT = sampleCount * upperTriangleBlockCount * square(blockSize<Int>) / omp_get_max_threads();
    // total number of comparisons in the inner loop per thread

    vector<tuple<size_t, size_t, double>> bestPairVector;
    bestPairVector.reserve(pairCount * omp_get_max_threads());

    std::atomic<size_t> nextBlockIndex1 = 0;

    // uint64_t ccc = clockCycleCount();

#pragma omp parallel
    {
        vector<tuple<size_t, size_t, double>> bestPairHeap;
        bestPairHeap.reserve(pairCount + 1);

        ArrayXXr<size_t> n0(blockSize<Int>, blockSize<Int>);
        ArrayXXr<size_t> n1(blockSize<Int>, blockSize<Int>);
        ArrayXXr<UInt> n(blockSize<Int>, blockSize<Int>);

        while (true) {
            const size_t blockIndex1 = nextBlockIndex1++;
            if (blockIndex1 >= blockCount)
                break;

            for (size_t blockIndex2 = blockIndex1; blockIndex2 != blockCount; ++blockIndex2) {

                n0 = 0;
                n = 0;
                UInt overflowGuard = 0;
                for (size_t i = 0; i != zeroSampleCount; ++i) {
                    ++overflowGuard;
                    if (overflowGuard == 0) {
                        n0 += n.cast<size_t>();
                        n = 0;
                        ++overflowGuard;
                    }
                    const Int* p1 = &zeroInData(i, blockIndex1 * blockSize<Int>);
                    const Int* p2 = &zeroInData(i, blockIndex2 * blockSize<Int>);
                    UInt* q = &n(0, 0);
                    processBlock_(p1, p2, q);
                }
                n0 += n.cast<size_t>();

                n1 = 0;
                n = 0;
                overflowGuard = 0;
                for (size_t i = 0; i != oneSampleCount; ++i) {
                    ++overflowGuard;
                    if (overflowGuard == 0) {
                        n1 += n.cast<size_t>();
                        n = 0;
                        ++overflowGuard;
                    }
                    const Int* p1 = &oneInData(i, blockIndex1 * blockSize<Int>);
                    const Int* p2 = &oneInData(i, blockIndex2 * blockSize<Int>);
                    UInt* q = &n(0, 0);
                    processBlock_(p1, p2, q);
                }
                n1 += n.cast<size_t>();

                // now n0(k1, k2) contains the number of samples i with
                //    outData(i) = 0
                // and
                //    inData(i, j1) > inData(i, j2)
                // where
                //   j1 = blockIndex1 * blockSize<Int> + k1
                //   j2 = blockIndex1 * blockSize<Int> + k2
                //
                // n1(k1, k2) contains the same but with outData(i) = 1

                for (size_t k1 = 0; k1 != blockSize<Int>; ++k1) {
                    const size_t j1 = blockIndex1 * blockSize<Int> + k1;

                    for (size_t k2 = 0; k2 != blockSize<Int>; ++k2) {
                        const size_t j2 = blockIndex2 * blockSize<Int> + k2;

                        const double prob0 = n0(k1, k2) * (1.0 / zeroSampleCount);
                        // probability that a sample i with outData(i) = 0 has inData(i, j1) > inData(i, j2)
                        const double prob1 = n1(k1, k2) * (1.0 / oneSampleCount);
                        // probability that a sample i with outData(i) = 1 has inData(i, j1) > inData(i, j2)
                        const double score = prob1 - prob0;

                        // const double a = static_cast<double>(zeroSampleCount - n0(k1, k2));
                        // const double b = static_cast<double>(n0(k1, k2));
                        // const double c = static_cast<double>(oneSampleCount - n1(k1, k2));
                        // const double d = static_cast<double>(n1(k1, k2));
                        // if (a + c < 3.0 || b + d < 3.0)
                        //    continue;
                        // const double score = (a * d - b * c) / std::sqrt((a + b) * (c + d) * (a + c) * (b + d));
                        // chi square statistic

                        const double lowestScoreInHeap = std::get<2>(bestPairHeap.front());
                        if (size(bestPairHeap) == pairCount && std::abs(score) <= lowestScoreInHeap)
                            continue;

                        const bool belowOrOnTheDiagonal = j1 >= j2;
                        const bool padding = j2 >= variableCount;
                        if (belowOrOnTheDiagonal || padding)
                            continue;

                        if (score >= 0)
                            bestPairHeap.push_back({j1, j2, score});
                        else
                            bestPairHeap.push_back({j2, j1, -score});

                        std::push_heap(begin(bestPairHeap), end(bestPairHeap), thirdGreater);

                        if (size(bestPairHeap) <= pairCount)
                            continue;

                        std::pop_heap(begin(bestPairHeap), end(bestPairHeap), thirdGreater);
                        bestPairHeap.pop_back();
                    }
                }
            }
        }

#pragma omp critical
        bestPairVector.insert(end(bestPairVector), begin(bestPairHeap), end(bestPairHeap));
    }

    // ccc = clockCycleCount() - ccc;
    // std::cout << "items / cc = " << static_cast<float>(ITEM_COUNT) / ccc << std::endl;

    ::pdqsort_branchless(begin(bestPairVector), end(bestPairVector), thirdGreater);
    ::shuffleEqual(begin(bestPairVector), end(bestPairVector), theRne, thirdGreater);

    ArrayXs variables1(pairCount);
    ArrayXs variables2(pairCount);
    ArrayXd scores(pairCount);
    for (size_t j = 0; j != pairCount; ++j)
        std::tie(variables1(j), variables2(j), scores(j)) = bestPairVector[j];
    return {variables1, variables2, scores};
};


void validateData_(CRefXXfr inData, CRefXu8 outData, size_t pairCount)
{
    const size_t variableCount = static_cast<size_t>(inData.cols());
    if (outData.rows() != inData.rows())
        throw std::invalid_argument("Indata and outdata have different numbers of samples.");
    if ((outData > 1).any())
        throw std::invalid_argument("Outdata has values that are not 0 or 1.");
    if (pairCount > variableCount * (variableCount - 1) / 2)
        throw std::invalid_argument("'PairCount' is larger than the total number of pairs.");
}


template<typename Int>
pair<ArrayXXr<Int>, ArrayXXr<Int>> preprocessData_(CRefXXfr inData, CRefXu8 outData, optional<CRefXs> samples)
{
    const size_t sampleCount = samples ? static_cast<size_t>(samples->rows()) : static_cast<size_t>(inData.rows());

    if (sampleCount == 0)
        throw std::invalid_argument("Sample count is zero.");

    vector<tuple<size_t, size_t, size_t>> sampleInfo;
    sampleInfo.reserve(sampleCount);

    size_t i0 = 0;
    size_t i1 = 0;
    if (samples) {
        for (size_t i : *samples) {
            if (outData(i) == 0) {
                sampleInfo.push_back({i, 0, i0});
                ++i0;
            }
            else {
                sampleInfo.push_back({i, 1, i1});
                ++i1;
            }
        }
    }
    else {
        for (size_t i = 0; i != sampleCount; ++i) {
            if (outData(i) == 0) {
                sampleInfo.push_back({i, 0, i0});
                ++i0;
            }
            else {
                sampleInfo.push_back({i, 1, i1});
                ++i1;
            }
        }
    }
    const size_t zeroSampleCount = i0;
    const size_t oneSampleCount = i1;

    if (zeroSampleCount == 0)
        throw std::invalid_argument("There are no samples with outdata 0.");
    if (oneSampleCount == 0)
        throw std::invalid_argument("There are no samples with outdata 1.");

    const size_t variableCount = static_cast<size_t>(inData.cols());
    const size_t blockCount = ::divideRoundUp(variableCount, blockSize<Int>);
    const size_t paddedVariableCount = blockCount * blockSize<Int>;

    ArrayXXr<Int> zeroInData(zeroSampleCount, paddedVariableCount);
    ArrayXXr<Int> oneInData(oneSampleCount, paddedVariableCount);

#pragma omp parallel private(i0, i1)
    {
        vector<pair<float, size_t>> tmp(variableCount);

        const size_t threadIndex = omp_get_thread_num();
        const size_t threadCount = omp_get_num_threads();

        const size_t kStart = sampleCount * threadIndex / threadCount;
        const size_t kStop = sampleCount * (threadIndex + 1) / threadCount;

        for (size_t k = kStart; k != kStop; ++k) {
            const size_t i = std::get<0>(sampleInfo[k]);
            if (std::get<1>(sampleInfo[k]) == 0) {
                i0 = std::get<2>(sampleInfo[k]);
                rankifyRow_<Int>(&inData.coeffRef(i, 0), &zeroInData(i0, 0), &tmp[0], variableCount);
            }
            else {
                i1 = std::get<2>(sampleInfo[k]);
                rankifyRow_<Int>(&inData.coeffRef(i, 0), &oneInData(i1, 0), &tmp[0], variableCount);
            }
        }
    }

    return pair{std::move(zeroInData), std::move(oneInData)};
}


template<typename Int>
inline void rankifyRow_(const float* inData, Int* rankData, pair<float, size_t>* tmp, size_t variableCount)
{
    const Int baseRank = numeric_limits<Int>::min();
    // the rank of the smallest value

    for (size_t j = 0; j != variableCount; ++j)
        tmp[j] = {inData[j], j};

    ::pdqsort_branchless(tmp, tmp + variableCount, ::firstLess);

    float prevVal = numeric_limits<float>::infinity();   // so that rank is not incremented for j = 0
    Int rank = baseRank;
    for (size_t j = 0; j != variableCount; ++j) {
        const float val = tmp[j].first;
        rank += val > prevVal;
        rankData[tmp[j].second] = rank;
        prevVal = val;
    }
}


#if USE_INTEL_INTRINSICS && defined(__AVX512F__) && defined(__AVX512BW__)

// not tested!

template<typename Int>
inline void processBlock_(const Int* __restrict p1, const Int* __restrict p2_, std::make_unsigned_t<Int>* __restrict q_)
{
    const __m512i one = mm512_set1<Int>(1);
    const size_t simdCount = sizeof(__m512i) / sizeof(Int);
    __m512i* q = reinterpret_cast<__m512i*>(q_);
    for (size_t k1 = 0; k1 != blockSize<Int>; ++k1) {
        const __m512i a = mm512_set1<Int>(*p1);
        const __m512i* p2 = reinterpret_cast<const __m512i*>(p2_);
        for (size_t k2 = 0; k2 != blockSize<Int> / simdCount; ++k2) {
            const typename mmask<Int>::type isLarger = mm512_cmpgt_mask<Int>(a, *p2);
            *q = mm512_mask_add<Int>(*q, isLarger, *q, one);
            ++p2;
            ++q;
        }
        ++p1;
    }
}

#elif USE_INTEL_INTRINSICS && defined(__AVX2__)

// MSVS does AVX2 autovectorize, but the manually vectorized code is about 20% faster (tested with MSVS 2019)
// (Inspection of the assembly code shows that the autovectorizer misses the trick
// of subtracting the return value from the comparison. Instead it takes bitwise and with 1 and then adds.)

template<typename Int>
inline void processBlock_(const Int* __restrict p1, const Int* __restrict p2_, std::make_unsigned_t<Int>* __restrict q_)
{
    const size_t simdCount = sizeof(__m256i) / sizeof(Int);
    __m256i* q = reinterpret_cast<__m256i*>(q_);
    for (size_t k1 = 0; k1 != blockSize<Int>; ++k1) {
        const __m256i a = mm256_set1<Int>(*p1);
        const __m256i* p2 = reinterpret_cast<const __m256i*>(p2_);
        for (size_t k2 = 0; k2 != blockSize<Int> / simdCount; ++k2) {
            const __m256i isLarger = mm256_cmpgt<Int>(a, *p2);
            // 0x0...0 if false and 0xf...f (i.e. -1) if true, hence we subtract
            *q = mm256_sub<Int>(*q, isLarger);
            ++p2;
            ++q;
        }
        ++p1;
    }
}

#else

template<typename Int>
inline void
processBlock_(const Int* __restrict p1, const Int* __restrict p2Start, std::make_unsigned_t<Int>* __restrict q)
{
    for (size_t k1 = 0; k1 != blockSize<Int>; ++k1) {
        const Int a = *p1;
        const Int* p2 = p2Start;
        for (size_t k2 = 0; k2 != blockSize<Int>; ++k2) {
            *q += a > *p2;
            ++p2;
            ++q;
        }
        ++p1;
    }
}

#endif

//----------------------------------------------------------------------------------------------------------------------

tuple<ArrayXs, ArrayXs, ArrayXd>
filterPairs(CRefXs variables1, CRefXs variables2, CRefXd scores, size_t singleVariableMaxFrequency)
{
    const size_t pairCount = static_cast<size_t>(variables1.rows());
    ASSERT(static_cast<size_t>(variables2.rows()) == pairCount);
    ASSERT(static_cast<size_t>(scores.rows()) == pairCount);

    const size_t variableCount = pairCount == 0 ? 0 : std::max(variables1.maxCoeff(), variables2.maxCoeff()) + 1;

    // pass 1

    vector<size_t> f(variableCount, 0);

    size_t kDest = 0;
    for (size_t kSrc = 0; kSrc != pairCount; ++kSrc) {
        const size_t j1 = variables1(kSrc);
        const size_t j2 = variables2(kSrc);
        if (f[j1] >= singleVariableMaxFrequency || f[j2] >= singleVariableMaxFrequency)
            continue;

        ++f[j1];
        ++f[j2];
        ++kDest;
    }
    const size_t filteredPairCount = kDest;

    // pass 2

    ArrayXs filteredVariables1(filteredPairCount);
    ArrayXs filteredVariables2(filteredPairCount);
    ArrayXd filteredScores(filteredPairCount);

    f.assign(variableCount, 0);

    kDest = 0;
    for (size_t kSrc = 0; kSrc != pairCount; ++kSrc) {
        const size_t j1 = variables1(kSrc);
        const size_t j2 = variables2(kSrc);
        if (f[j1] >= singleVariableMaxFrequency || f[j2] >= singleVariableMaxFrequency)
            continue;

        filteredVariables1(kDest) = j1;
        filteredVariables2(kDest) = j2;
        filteredScores(kDest) = scores(kSrc);

        ++f[j1];
        ++f[j2];
        ++kDest;
    }
    ASSERT(kDest == filteredPairCount);

    return {std::move(filteredVariables1), std::move(filteredVariables2), std::move(filteredScores)};
}

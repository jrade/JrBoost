//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


template<typename T>
constexpr T square(const T& a)
{
    return a * a;
}

template<typename T>
constexpr T divideRoundUp(const T& a, const T& b)
{
    return (a + b - static_cast<T>(1)) / b;
}

//----------------------------------------------------------------------------------------------------------------------

// Compile time integer square root by Kim Walisch
// Returns the square root rounded down to the nearest integer
// See constexpr_sqrt.cpp at gist.github.com/kimwalisch/d249cf684a58e1d892e1

constexpr size_t integerSqrtImpl_(size_t x, size_t lo, size_t hi)
{
    size_t mid = (lo + hi + 1) / 2;
    return lo == hi ? lo : ((x / mid < mid) ? integerSqrtImpl_(x, lo, mid - 1) : integerSqrtImpl_(x, mid, hi));
}

constexpr size_t integerSqrt(size_t x) { return integerSqrtImpl_(x, 0, x / 2 + 1); }

//----------------------------------------------------------------------------------------------------------------------

constexpr auto firstLess = [](const auto& x, const auto& y) { return std::get<0>(x) < std::get<0>(y); };

constexpr auto secondLess = [](const auto& x, const auto& y) { return std::get<1>(x) < std::get<1>(y); };

constexpr auto thirdLess = [](const auto& x, const auto& y) { return std::get<2>(x) < std::get<2>(y); };

constexpr auto firstGreater = [](const auto& x, const auto& y) { return std::get<0>(x) > std::get<0>(y); };

constexpr auto secondGreater = [](const auto& x, const auto& y) { return std::get<1>(x) > std::get<1>(y); };

constexpr auto thirdGreater = [](const auto& x, const auto& y) { return std::get<2>(x) > std::get<2>(y); };

//----------------------------------------------------------------------------------------------------------------------

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

inline uint64_t clockCycleCount()
{
    __faststorefence();
    uint64_t t = __rdtsc();
    __faststorefence();
    return t;
}

//----------------------------------------------------------------------------------------------------------------------

// Wrapper class that makes instances aligned to cache lines
// This prevents false sharing between threads.

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)   // 'struct_name' : structure was padded due to __declspec(align())
#endif

template<typename T>
class alignas(std::hardware_destructive_interference_size) CacheLineAligned : public T {
public:
    template<typename... Values>
    CacheLineAligned(Values... values) : T(std::forward<Values>(values)...)
    {
    }
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

//----------------------------------------------------------------------------------------------------------------------

// Alternative to std::make_unique that works with classes with protected or private constructors
// With private constructors, MakeUniqueHelper<T> must be declared a friend of T

template<typename T>
class MakeUniqueHelper : public T {
public:
    template<typename... Values>
    MakeUniqueHelper(Values... values) : T(std::forward<Values>(values)...)
    {
    }
};

template<typename T, typename... Values>
unique_ptr<T> makeUnique(Values... values)
{
    return std::make_unique<MakeUniqueHelper<T>>(std::forward<Values>(values)...);
}

//----------------------------------------------------------------------------------------------------------------------

// Alternative to std::make_shared that works with classes with protected or private constructors
// With private constructors, MakeSharedHelper<T> must be declared a friend of T


template<typename T>
class MakeSharedHelper : public T {
public:
    template<typename... Values>
    MakeSharedHelper(Values... values) : T(std::forward<Values>(values)...)
    {
    }
};

template<typename T, typename... Values>
shared_ptr<T> makeShared(Values... values)
{
    return std::make_shared<MakeSharedHelper<T>>(std::forward<Values>(values)...);
}

//----------------------------------------------------------------------------------------------------------------------

class AssertionError : public std::logic_error {
public:
    AssertionError(const string& condition, const string& file, int line) : logic_error(message_(condition, file, line))
    {
    }

private:
    static string message_(const string& condition, const string& file, int line)
    {
        stringstream ss;
        ss << "\nCondition: " << condition << "\nFile: " << file << "\nLine: " << line;
        return ss.str();
    }
};

template<typename T>
inline void assert_(const T& ok, const char* condition, const char* file, int line)
{
    if (!ok)
        throw AssertionError(condition, file, line);
}

#define ASSERT(A) assert_((A), #A, __FILE__, __LINE__)

//----------------------------------------------------------------------------------------------------------------------

inline void parseError [[noreturn]] (istream& is)
{
    string msg = "Not a valid JrBoost predictor file.";

    std::streampos pos = is.tellg();
    if (pos != static_cast<std::streampos>(-1))
        msg += "\n(Parsing error after " + std::to_string(pos) + " bytes)";

    throw std::runtime_error(msg);
}

//----------------------------------------------------------------------------------------------------------------------

// Fills the range that starts at q0 with a permutation
// i0, ..., i{n-1} of 0, 1, ...., n-1 such that
// f(p0[i0]) <= f(p0[i1]) <= ... <= f(p0[i{n-1}]).
// Here n is the size of the range [p0, p1).

template<typename T, typename U, typename F>
inline void sortedIndices(T p0, T p1, U q0, F f)
{
    using R = std::decay_t<decltype(f(*p0))>;
    using N = std::decay_t<decltype(*q0)>;

    size_t n = p1 - p0;
    vector<pair<N, R>> tmp(n);
    auto r0 = begin(tmp);
    auto r1 = end(tmp);

    N i = static_cast<N>(0);
    auto p = p0;
    auto r = r0;
    while (p != p1)
        *(r++) = std::make_pair(i++, f(*(p++)));

    pdqsort_branchless(r0, r1, ::secondLess);

    r = r0;
    auto q = q0;
    while (r != r1)
        *(q++) = (r++)->first;
}

//----------------------------------------------------------------------------------------------------------------------

// shuffles each block of equal equal elements in a sorted range

template<typename It, typename Rng, typename Comp>
void shuffleEqual(It p, It pEnd, Rng& rng, Comp lessThan)
{
    if (p == pEnd)
        return;
    It q = p;
    ++q;
    while (q != pEnd) {
        if (!lessThan(*p, *q)) {
            do
                ++q;
            while (q != pEnd && !lessThan(*p, *q));
            std::shuffle(p, q, rng);
            if (q == pEnd)
                break;
        }
        p = q;
        ++q;
    }
}

//----------------------------------------------------------------------------------------------------------------------

using RandomNumberEngine = splitmix;

class InitializedRandomNumberEngine : public RandomNumberEngine {
public:
    InitializedRandomNumberEngine()
    {
        std::random_device rd;
        seed(rd);
    }
};

inline thread_local InitializedRandomNumberEngine theRne;

//----------------------------------------------------------------------------------------------------------------------

class InterruptHandler {
public:
    virtual void check() = 0;
};

inline InterruptHandler* currentInterruptHandler = nullptr;

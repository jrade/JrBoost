//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


template<typename T>
T square(const T& a)
{
    return a * a;
}

template<typename T>
inline T divideRoundUp(const T& a, const T& b)
{
    return (a + b - static_cast<T>(1)) / b;
}

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
// With private constructors, MakeUniqueHelper<T> must be declared a friend´of T

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
// With private constructors, MakeSharedHelper<T> must be declared a friend´of T


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

    pdqsort_branchless(r0, r1, [](const auto& x, const auto& y) { return x.second < y.second; });

    r = r0;
    auto q = q0;
    while (r != r1)
        *(q++) = (r++)->first;
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

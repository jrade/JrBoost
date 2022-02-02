//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class FastBernoulliDistribution {
public:
    FastBernoulliDistribution(uint64_t m, uint64_t n) : m_(static_cast<double>(m)), n_(static_cast<double>(n))
    {
        // ASSERT(0 <= m && m <= n);
    }

    template<typename R>
    bool operator()(R& r)
    {
        double a = static_cast<double>(r() - R::min());
        constexpr double b = static_cast<double>(R::max() - R::min()) * (1.0 + 2.0 * numeric_limits<double>::epsilon());
        return a * n_ < b * m_;
    }

private:
    double m_;
    double n_;
};

// The following three statements are true
//     (1) returns true with probability approximately m / n
//     (2) always returns false for m = 0
//     (3) always returns true for m = n

// That (1) and (2) hold should be obvious.
//
// That (3) holds follows from the inequality
//    a * b  < a * (b * (1 + 2 * e))
// for any normal (i.e. not denormal) positive a and b with 2 * a * b < infinity.
// Here e = numeric_limits<double>::epsilon().
//
// This inequality can be seen as follows.
// Since (2^m * a) * b = 2^m * (a * b) we may assume that 1 <= a < 2 and 1 <= b < 2.
// Then
//    a * (b * (1 + 2e)) >= a * (b + 2e) >= a * b + 2e
// Now
//    a * b <= (2 - e) * (2 - e) = 4 - 4 * e < 4
// so
//    a * b + 2 * e > a * b

//----------------------------------------------------------------------------------------------------------------------

// This class is only safe if n and R:max() are < (1LL << 32).
// Otherwise the products can overflow.

class VeryFastBernoulliDistribution {
public:
    VeryFastBernoulliDistribution(uint64_t m, uint64_t n) : m_(m), n_(n)
    {
        // ASSERT(0 <= m && m <= n);
    }

    template<typename R>
    bool operator()(R& r)
    {
        uint64_t a = r() - R::min();
        constexpr uint64_t b = static_cast<uint64_t>(R::max() - R::min()) + 1;
        return a * n_ < b * m_;
    }

private:
    uint64_t m_;
    uint64_t n_;
};

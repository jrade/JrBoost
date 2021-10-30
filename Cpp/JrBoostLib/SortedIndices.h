//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

template<typename T, typename U, typename F>
inline void sortedIndices(T p0, T p1, U q0, F f)
{
    using R = decltype(f(*p0));
    static_assert(std::is_arithmetic<R>::value);

    vector<pair<R, size_t>> tmp(p1 - p0);
    auto r0 = begin(tmp);
    auto r1 = end(tmp);

    size_t i = 0;
    auto p = p0;
    auto r = r0;
    while (p != p1)
        *(r++) = std::make_pair(f(*(p++)), i++);

    pdqsort_branchless(
        r0,
        r1,
        [](const auto& x, const auto& y) { return x.first < y.first; }
    );

    r = r0;
    auto q = q0;
    while (r != r1)
        *(q++) = (r++)->second;
}

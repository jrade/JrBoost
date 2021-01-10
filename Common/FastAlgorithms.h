#pragma once

// The following algorithms have branch-free implementations
// This gives a huge speedup compared with the STL algorithms

//----------------------------------------------------------------------------------------------------------------------

// requirements: T is a random access iterator type, R is a random mumber engine
// precondition: see the ASSERT statements
// postcondition: n randomly selected elements in the range [p0, p1) are = 1, the remaining are = 0

template<typename T, typename R>
inline void fastRandomMask(T p0, T p1, uint64_t n, R& r)
{
	ASSERT(0 <= n && p1 >= p0 && n <= static_cast<uint64_t>(p1 - p0));

	if (static_cast<uint64_t>(R::max() - R::min()) <= std::numeric_limits<uint32_t>::max()
		&& static_cast<uint64_t>(p1 - p0) <= std::numeric_limits<uint32_t>::max()) {
		while (n != 0) {
			uint64_t a = static_cast<uint64_t>(r() - R::min());
			constexpr uint64_t b = static_cast<uint64_t>(R::max() - R::min()) + 1;
			bool m = a * static_cast<uint64_t>(p1 - p0) < b * n;
			*p0 = m;
			++p0;
			n -= m;
		}
	}
	else {
		// uint64_t products might overflow so use double instead
		while (n != 0) {
			double a = static_cast<double>(r() - R::min());
			constexpr double b = static_cast<double>(R::max() - R::min());
			bool m = a * static_cast<double>(p1 - p0) <= b * static_cast<double>(n);
			*p0 = m;
			++p0;
			n -= m;
		}
	}
	while (p0 != p1) {
		*p0 = 0;
		++p0;
	}
}

// The following two statements are true
//     (1) m = true with probability approximately n / (p1 - p0)
//     (2) m always = true when n = p1 - p0 > 0
// That
//     (3) m always = false when n = 0
// holds in the size_t version but can fail in the the double version.
// The code relies on (1) and (2) but not on (3).

//----------------------------------------------------------------------------------------------------------------------

// Note: If we have IEEE754 compliant double arithmetic both at compile and run time and we define m as
//
//      double a = static_cast<double>(r() - R::min());
//      constexpr double b = static_cast<double>(R::max() - R::min())
//          * (1 + 2 * std::numeric_limits::epsilon<double>());
//      bool m = a * static_cast<double>(p1 - p0) < b * static_cast<double>(n);
//
// then (1), (2) and (3) hold in the double case.
// That (3) holds follows from the inequality
//    a * b  < a * (b * (1 + 2 *e))
// for any normal (i.e. not denormal) positive a and b with 2 * a * b < infinity.
// Here e = std::numeric_limits<double>::epsilon().
//
// This inequality can be seen as follows.
// Since (2^k * a) * b = 2^k * (a * b) we may assume that 1 <= a < 2 and 1 <= b < 2.
// Then
//    a * (b * (1 + 2e)) >= a * (b + 2e) >= a * b + 2e
// Now 
//    a * b <= (2 - e) * (2 - e) = 4 - 4 * e < 4
// so
//    a * b + 2 * e > a * b

//----------------------------------------------------------------------------------------------------------------------

// requirements: T is a random access iterator type, R is a random mumber engine
// precondition: see the ASSERT statements
// postcondition: the range [q0, q1) contains a random subset of the elements in [p0, p1)
// the order of the elements is preserved.

template<typename T, typename R>
inline void fastOrderedRandomSubset(T p0, T p1, T q0, T q1, R& r)
{
	ASSERT(q0 <= q1 && p0 <= p1);
	ASSERT(q1 - q0 <= p1 - p0);
	ASSERT(q0 <= p0 || q0 >= p1);

	if (static_cast<uint64_t>(R::max() - R::min()) <= std::numeric_limits<uint32_t>::max()
		&& static_cast<uint64_t>(p1 - p0) <= std::numeric_limits<uint32_t>::max()) {
		while (q0 != q1) {
			*q0 = *p0;
			uint64_t a = static_cast<uint64_t>(r() - R::min());
			constexpr uint64_t b = static_cast<uint64_t>(R::max() - R::min()) + 1;
			bool m =  a * static_cast<uint64_t>(p1 - p0) <  b * static_cast<uint64_t>(q1 - q0);
			q0 += m;
			++p0;
		}
	}
	else {
		// uint64_t products might overflow so use double instead
		while (q0 != q1) {
			*q0 = *p0;
			double a = static_cast<double>(r() - R::min());
			constexpr double b = static_cast<double>(R::max() - R::min());
			bool m = a * static_cast<double>(p1 - p0) <= b * static_cast<double>(q1 - q0);
			q0 += m;
			++p0;
		}
	}
}

// see the discussion of m above

//----------------------------------------------------------------------------------------------------------------------

template<typename T, typename U, typename P>
inline T fastCopyIf(T p0, U q0, U q1, P pred)
{
	while (q0 != q1) {
		auto val = *p0;
		*q0 = val;
		++p0;
		q0 += pred(val);
	}
	return p0;
}

// Why not inline T fastCopyIf(T p0, Tp1, U q0, P pred)?
// Because then we  might write at q1, i.e. after the end of the range [q0, q1)

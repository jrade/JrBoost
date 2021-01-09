#pragma once


// requirements: T is a random access iterator type, R is a random mumber engine
// precondition: see the ASSERT statements
// postcondition: the range [q0,q1) contains a random subset of the elements in [p0,p1)
// the order of the elements is preserved.

template<typename T, typename R>
inline void orderedRandomSubset(T p0, T p1, T q0, T q1, R& r)
{
	ASSERT(0 <= q1 - q0 && q1 - q0 <= p1 - p0);
	ASSERT(q0 <= p0 || q0 >= p1);

	// we look at each element in the input (p0) and decide whether it should be included in the output (q0)
	// it is inluded with probability
	// (q1 - q0) / (p1 - p0) = (number of slots left to fill) / (number of elements left to pick from)

	if (R::max() - R::min() < 0x100000000 && p1 - p0 < 0x100000000) {
		while (q0 < q1) {
			*q0 = *p0;
			bool m = static_cast<uint64_t>(r() - R::min()) * static_cast<uint64_t>(p1 - p0)
				< (static_cast<uint64_t>(R::max() - R::min()) + 1) * static_cast<uint64_t>(q1 - q0);
			// m = true with probability (q1 - q0) / (p1 - p0)
			// i.e. (number of positions left to fill) / (number of elements left to pick from)
			// It is guranteed that m = true when q1 - q0  = p1 - p0
			// and that m = false when q1 - q0 = 0
			q0 += m;
			++p0;
		}
	}
	else {
		// uint64_t products might overflow so use double instead
		while (q0 < q1) {
			*q0 = *p0;
			bool m = static_cast<double>(r() - R::min()) * static_cast<double>(p1 - p0)
				<= static_cast<double>(R::max() - R::min()) * static_cast<double>(q1 - q0);
			// m = true with probability (q1 - q0) / (p1 - p0)
			// i.e. (number of positions left to fill) / (number of elements left to pick from)
			// It is guranteed that m = true when q1 - q0  = p1 - p0
			// It is NOT guaranteed that m = false when q1 - q0 = 0
			// (If we were to replace static_cast<double>(R::max() - R::min()) 
			// by (static_cast<double>(R::max() - R::min()) + 1)
			// then it will be guaranteed that  m = false when q1 - q0 = 0 but we can no longer guarantee that
			// m = true when q1 - q0  = p1 - p0 due to rounding.)
			q0 += m;
			++p0;
		}
	}
}


// requirements: T is a random access iterator type, R is a random mumber engine
// precondition: see the ASSERT statements
// postcondition: n randomly selected elements in the range [p0,p1) are = 1, the remaining are = 0

template<typename T, typename R>
inline void randomMask(T p0, T p1, int64_t n, R& r)
{
	ASSERT(0 <= n && n <= p1 - p0);

	if (R::max() - R::min() < 0x100000000 && p1 - p0 < 0x100000000) {
		while (p0 < p1) {
			bool m = static_cast<uint64_t>(r() - R::min()) * static_cast<uint64_t>(p1 - p0)
				< (static_cast<uint64_t>(R::max() - R::min()) + 1) * static_cast<uint64_t>(n);
			// m = true with probability n / (p1 - p0)
			// It is guaranteed that m = true when n = p1 - p0
			// and that m = false when n = 0
			*p0 = m;
			n -= m;
			++p0;
		}
		ASSERT(n == 0);
	}
	else {
		// uint64_t products might overflow so use double instead
		while (n > 0) {
			bool m = static_cast<double>(r() - R::min()) * static_cast<double>(p1 - p0)
				<= static_cast<double>(R::max() - R::min()) * static_cast<double>(n);
			// m = true with probability n / (p1 - p0)
			// It is guaranteed that m = true when n = p1 - p0
			// It is NOT guarantedd that m = false when n = 0
			*p0 = m;
			n -= m;
			++p0;
		}
		std::fill(p0, p1, char{ 0 });
	}
}


template<typename T, typename U, typename P>
inline T copyIf(T p0, U q0, U q1, P pred)
{
	while (q0 != q1) {
		auto val = *p0;
		*q0 = val;
		++p0;
		q0 += pred(val);
	}
	return p0;
}

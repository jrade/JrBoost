#pragma once

#include "FastBernoulliDistribution.h"

#define FAST_BERNOULLI

// The following algorithms have branch-free implementations
// This gives a huge speedup compared with the STL algorithms


//----------------------------------------------------------------------------------------------------------------------

// requirements: T is a random access iterator type, R is a random mumber engine
// precondition: see the ASSERT statements
// postcondition: k randomly selected elements in the range [p0, p1) are = 1, the remaining are = 0

template<typename T, typename R>
inline void fastRandomMask(T p0, T p1, size_t k, R& r)
{
	ASSERT(0 <= k && p0 <= p1 && k <= static_cast<size_t>(p1 - p0));

	while (p0 != p1) {
#ifdef FAST_BERNOULLI
		bool m = FastBernoulliDistribution<size_t>(k, p1 - p0) (r);
#else
		bool m = std::bernoulli_distribution(
			static_cast<double>(k) 
			/ static_cast<double>(p1 - p0)
		)(r);
#endif
		*p0 = m;
		++p0;
		k -= m;
	}
}

//----------------------------------------------------------------------------------------------------------------------

// requirements: T is a random access iterator type, R is a random mumber engine
// precondition: see the ASSERT statements
// postcondition: k randomly selected elements in the range [p0, p1) are = 1, the remaining are = 0

template<typename T, typename U, typename V, typename W, typename R>
inline void fastStratifiedRandomMask(
	T p0,
	T p1,
	U q0,
	V sampleCountByStratum,
	W subsampleCountByStratum,
	R& r
)
{
	// assert p1 - p0 = sum sampleCountByStratum

	while (p0 != p1) {
		size_t stratum = *p0;
#ifdef FAST_BERNOULLI
		bool m = FastBernoulliDistribution<size_t>(
			subsampleCountByStratum[stratum],
			sampleCountByStratum[stratum]
		)(r);
#else
		bool m = std::bernoulli_distribution(
			static_cast<double>(subsampleCountByStratum[stratum])
			/ static_cast<double>(sampleCountByStratum[stratum])
		)(r);
#endif
		*q0 = m;
		++p0;
		++q0;
		--sampleCountByStratum[stratum];
		subsampleCountByStratum[stratum] -= m;
	}
}

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

	while (q0 != q1) {
		*q0 = *p0;
#ifdef FAST_BERNOULLI
		bool m = FastBernoulliDistribution<size_t>(q1 - q0, p1 - p0)(r);
#else
		bool m = std::bernoulli_distribution(
			static_cast<double>(q1 - q0)
			/ static_cast<double>(p1 - p0)
		)(r);
#endif
		q0 += m;
		++p0;
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

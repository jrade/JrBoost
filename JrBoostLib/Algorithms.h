#pragma once

// requirements: T is a random access iterator type, R is a random mumber generator type with range [0, 2^32)
// precondition: q1 - q0 <= p1 - p0
// precondition: if the two ranges overlap, then q0 <= p0
// postcondition: the range [q0,q1) contains a random subset of the elements in [p0,p1)
// note: the order of the elements is preserved. In particular, if [p0,p1) is sorted, then [q0,q1) will be sorted

template<typename T, typename R>
inline void orderedRandomSubset(T p0, T p1, T q0, T q1, R& r)
{
	// check that the arguments are valid
	ASSERT(p1 >= p0 && q1 >= q0 && q1 - q0 <= p1 - p0);
	ASSERT(q0 <= p0 || q0 >= p1);

	// check that the products will not overflow
	static_assert(sizeof(R::result_type) == 4);
	ASSERT(p1 - p0 < 0x100000000);

	// we look at each element in the input (p0) and decide whether it should be included in the output (q0)
	// it is inluded with probability
	// (q1 - q0) / (p1 - p0) = (number of slots left to fill) / (number of elements left to pick from)

	while (q0 < q1) {
		*q0 = *p0;
		int64_t a = r() - R::min();
		int64_t b = static_cast<int64_t>(R::max()) - R::min() + 1;
		bool m = (a * (p1 - p0) < b * (q1 - q0));
		q0 += m;    // this is a hack to get branch free code
		++p0;
	}
	if (p0 > p1)
		cout << "b" << endl;
}

template<typename T, typename R>
inline void randomMask(T p0, T p1, int64_t n, R& r)
{
	// check that the arguments are valid
	ASSERT(p0 <= p1);
	ASSERT(n >= 0 && n <= p1 - p0);

	// make sure that the products will not overflow
	static_assert(sizeof(R::result_type) == 4);
	ASSERT(p1 - p0 < 0x100000000);

	// we look at each element in the input (p0) and decide whether it should be included in the output (q0)
	// it is inluded with probability
	// *p0 is set to 1 with probability n / (p1 - p0)

	while (p0 < p1) {
		int64_t a = r() - R::min();
		int64_t b = static_cast<int64_t>(R::max()) - R::min() + 1;
		bool m = (a * (p1 - p0) < b * n);
		*p0 = m;
		n -= m;
		++p0;
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

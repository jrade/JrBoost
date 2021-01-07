#pragma once

using RNE = std::mt19937;		// random number engine

inline RNE& theRNE()
{
	static RNE theRNE_(static_cast<RNE::result_type>(time(0)));
	return theRNE_;
}

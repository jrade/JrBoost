#pragma once

#include "Gerstmann.h"

using RNE = splitmix;

inline static std::random_device theRandomDevice;

inline static RNE theRNE(theRandomDevice);

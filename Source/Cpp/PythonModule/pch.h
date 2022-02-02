//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

// precompiled header file

#pragma once

#include "../Common.h"


// pybind11

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

inline const char* thePybind11Version
    = STR(PYBIND11_VERSION_MAJOR) "." STR(PYBIND11_VERSION_MINOR) "." STR(PYBIND11_VERSION_PATCH);

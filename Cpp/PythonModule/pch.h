// precompiled header file

#pragma once

#include "../Common.h"


// pybind11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

inline const char* pybind11Version = STR(PYBIND11_VERSION_MAJOR) "." STR(PYBIND11_VERSION_MINOR) "." STR(PYBIND11_VERSION_PATCH);

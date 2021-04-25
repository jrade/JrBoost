//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "../JrBoostLib/InterruptHandler.h"


class PyInterruptHandler : public InterruptHandler {
public:
    virtual void check()
    {
        namespace py = pybind11;
        py::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() != 0)
            throw py::error_already_set();
    }
};

inline PyInterruptHandler thePyInterruptHandler;

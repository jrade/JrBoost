//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class PyInterruptHandler : public InterruptHandler {
public:
    virtual void check()
    {
        if (std::this_thread::get_id() != theMainThreadId)
            return;

        uint64_t ccc = __rdtsc();
        if (ccc >= lastCcc_ && ccc < lastCcc_ + 1.0e8)
            return;
        lastCcc_ = ccc;

        pybind11::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() == 0)
            return;

        throw pybind11::error_already_set();
    }

private:
    uint64_t lastCcc_ = 0;
};

inline PyInterruptHandler thePyInterruptHandler;

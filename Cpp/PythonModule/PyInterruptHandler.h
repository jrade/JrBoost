//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once


class PyInterruptHandler : public InterruptHandler {
public:
    virtual void check()
    {
        if (std::this_thread::get_id() != theMainThreadId)
            return;

        time_t presentTime = time(nullptr);
        if (presentTime < lastTime_ + 1)
            return;
        lastTime_ = presentTime;

        pybind11::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() == 0)
            return;

        throw py::error_already_set();
    }

private:
    time_t lastTime_ = 0;
};

inline PyInterruptHandler thePyInterruptHandler;

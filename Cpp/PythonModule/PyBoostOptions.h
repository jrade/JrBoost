//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "../JrBoostLib/BoostOptions.h"

namespace py = pybind11;


namespace pybind11::detail
{
    // custom converter for class BoostOptions

    template<> struct type_caster<BoostOptions> {
    public:
        PYBIND11_TYPE_CASTER(BoostOptions, _("BoostOptions"));
        bool load(py::handle h, bool);
        static py::handle cast(const BoostOptions& opt, py::return_value_policy, py::handle /*parent*/);

    private:
        using PyBoostOptions_ = map<string, std::variant<bool, size_t, double>>;
        static BoostOptions fromPython_(const PyBoostOptions_& param);
        static PyBoostOptions_ toPython_(const BoostOptions& opt);
    };
}

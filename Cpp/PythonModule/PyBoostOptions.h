//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "../JrBoostLib/BoostOptions.h"


namespace pybind11::detail {

// custom converter for class BoostOptions

template<>
struct type_caster<BoostOptions> {
public:
    PYBIND11_TYPE_CASTER(BoostOptions, _("BoostOptions"));
    bool load(pybind11::handle h, bool);
    static pybind11::handle cast(const BoostOptions& opt, pybind11::return_value_policy, pybind11::handle /*parent*/);

private:
    using PyBoostOptions_ = map<string, std::variant<bool, size_t, double>>;
    static BoostOptions fromPython_(const PyBoostOptions_& param);
    static PyBoostOptions_ toPython_(const BoostOptions& opt);
};

}   // namespace pybind11::detail

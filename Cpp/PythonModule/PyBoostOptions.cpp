//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "PyBoostOptions.h"

namespace py = pybind11;
using tcBO = py::detail::type_caster<BoostOptions>;

// The conversions are done with the class
// PyBoostOptions_ = map<string, variant<bool, size_t, double>>
// as an intermediate step


// Conversion from Python to C++ ---------------------------------------------------------------------------------------

// This conversion may fail
// Take care to provide clear error messages
// By throwing exceptions instead of returning false, the user gets better error messages
// (This also means that PyBind11 will not be able to handle muliple overloads.)

bool tcBO::load(py::handle h, bool)
{
    PyBoostOptions_ pyOpt;
    try {
        pyOpt = h.cast<PyBoostOptions_>();
    }
    catch (const py::cast_error&) {
        string msg = "Boost parameters must be a dict "
            "with keys of type str and values of type bool, int or float.";
        throw std::invalid_argument(msg);
    }

    value = fromPython_(pyOpt);
    // The macro PYBIND11_TYPE_CASTER defines a data member 'value' of type BoostOptions
    return true;
}

BoostOptions tcBO::fromPython_(const PyBoostOptions_& pyOpt)
{
    BoostOptions opt;

    for (const auto& [key, value] : pyOpt) {
        try {
            if (key == "gamma")
                opt.setGamma(std::get<double>(value));
            else if (key == "iterationCount")
                opt.setIterationCount(std::get<size_t>(value));
            else if (key == "eta")
                opt.setEta(std::get<double>(value));
            else if (key == "fastExp")
                opt.setFastExp(std::get<bool>(value));
            else if (key == "maxDepth")
                opt.setMaxDepth(std::get<size_t>(value));
            else if (key == "usedSampleRatio")
                opt.setUsedSampleRatio(std::get<double>(value));
            else if (key == "usedVariableRatio")
                opt.setUsedVariableRatio(std::get<double>(value));
            else if (key == "topVariableCount")
                opt.setTopVariableCount(std::get<size_t>(value));
            else if (key == "minAbsSampleWeight")
                opt.setMinAbsSampleWeight(std::get<double>(value));
            else if (key == "minRelSampleWeight")
                opt.setMinRelSampleWeight(std::get<double>(value));
            else if (key == "minNodeSize")
                opt.setMinNodeSize(std::get<size_t>(value));
            else if (key == "minNodeWeight")
                opt.setMinNodeWeight(std::get<double>(value));
            else if (key == "minGain")
                opt.setMinGain(std::get<double>(value));
            else if (key == "isStratified")
                opt.setIsStratified(std::get<bool>(value));
            else if (key == "pruneFactor")
                opt.setPruneFactor(std::get<double>(value));
            else {
                string msg = "The key '" + key + "' is not a valid boost parameters key.";
                throw std::invalid_argument(msg);
            }
        }
        catch (const std::bad_variant_access&) {
            string msg = "The boost parameters key '" + key + "' has value of wrong type.";
            throw std::invalid_argument(msg);
        }
    }

    return opt;
}


// Conversion from C++ to Python ---------------------------------------------------------------------------------------

// This conversion never fails
// Be careful with the reference count of the newly created Python object

py::handle tcBO::cast(const BoostOptions& opt, py::return_value_policy, py::handle /*parent*/)
{
    PyBoostOptions_ pyOpt = toPython_(opt);
    py::object obj = py::cast(pyOpt);
    py::handle h = obj.release();
    return h;
}

tcBO::PyBoostOptions_ tcBO::toPython_(const BoostOptions& opt)
{
    PyBoostOptions_ pyOpt;

    pyOpt["gamma"] = opt.gamma();
    pyOpt["iterationCount"] = opt.iterationCount();
    pyOpt["eta"] = opt.eta();
    pyOpt["fastExp"] = opt.fastExp();
    pyOpt["maxDepth"] = opt.maxDepth();
    pyOpt["usedSampleRatio"] = opt.usedSampleRatio();
    pyOpt["usedVariableRatio"] = opt.usedVariableRatio();
    pyOpt["topVariableCount"] = opt.topVariableCount();
    pyOpt["minAbsSampleWeight"] = opt.minAbsSampleWeight();
    pyOpt["minRelSampleWeight"] = opt.minRelSampleWeight();
    pyOpt["minNodeSize"] = opt.minNodeSize();
    pyOpt["minNodeWeight"] = opt.minNodeWeight();
    pyOpt["minGain"] = opt.minGain();
    pyOpt["isStratified"] = opt.isStratified();
    pyOpt["pruneFactor"] = opt.pruneFactor();

    return pyOpt;
}

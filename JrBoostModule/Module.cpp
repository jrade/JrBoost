#include "pch.h"

#include "../JrBoostLib/StumpOptions.h"
#include "../JrBoostLib/StumpPredictor.h"
#include "../JrBoostLib/StumpTrainer.h"

#include "../JrBoostLib/BoostOptions.h"
#include "../JrBoostLib/BoostPredictor.h"
#include "../JrBoostLib/BoostTrainer.h"

namespace py = pybind11;


PYBIND11_MODULE(jrboost, mod)
{
    py::register_exception<AssertionError>(mod, "AssertionError", PyExc_AssertionError);


    // Stump classes

    py::class_<StumpPredictor>{ mod, "StumpPredictor" }
        .def("variableCount", &StumpPredictor::variableCount)
        .def("predict", &StumpPredictor::predict, py::arg().noconvert());

    py::class_<StumpOptions>{ mod, "StumpOptions" }
        .def(py::init<>())
        .def_property("usedSampleRatio", &StumpOptions::usedSampleRatio, &StumpOptions::setUsedSampleRatio)
        .def_property("usedVariableRatio", &StumpOptions::usedVariableRatio, &StumpOptions::setUsedVariableRatio)
        .def_property("topVariableCount", &StumpOptions::topVariableCount, &StumpOptions::setTopVariableCount)
        .def_property("minNodeSize", &StumpOptions::minNodeSize, &StumpOptions::setMinNodeSize)
        .def_property("minNodeWeight", &StumpOptions::minNodeWeight, &StumpOptions::setMinNodeWeight)
        .def_property("isStratified", &StumpOptions::isStratified, &StumpOptions::setIsStratified)
        .def_property("profile", &StumpOptions::profile, &StumpOptions::setProfile);

    py::class_<StumpTrainer>{ mod, "StumpTrainer" }
        .def(py::init<RefXXf, RefXs>(), py::arg().noconvert(), py::arg().noconvert())
        .def("train", &StumpTrainer::train);


    // Boost classes

    py::class_<BoostPredictor>{ mod, "BoostPredictor" }
        .def("variableCount", &BoostPredictor::variableCount)
        .def("predict", &BoostPredictor::predict, py::arg().noconvert());

    py::class_<BoostOptions> opt{ mod, "BoostOptions" };
    opt.def(py::init<>())
        .def_property("method", &BoostOptions::method, &BoostOptions::setMethod)
        .def_property("iterationCount", &BoostOptions::iterationCount, &BoostOptions::setIterationCount)
        .def_property("eta", &BoostOptions::eta, &BoostOptions::setEta)
        .def_property("logStep", &BoostOptions::logStep, &BoostOptions::setLogStep)
        .def_property_readonly(
            "base",
            static_cast<StumpOptions& (BoostOptions::*)()>(&BoostOptions::base)
        );
    py::enum_<BoostOptions::Method>(opt, "Method")
        .value("Ada", BoostOptions::Method::Ada)
        .value("Logit", BoostOptions::Method::Logit);

    py::class_<BoostTrainer>{ mod, "BoostTrainer" }
        .def(py::init<RefXXf, RefXs>(), py::arg().noconvert(), py::arg())
        .def("train", &BoostTrainer::train)
        .def("trainAndEval", &BoostTrainer::trainAndEval)
        .def_readwrite_static("threadCount", &BoostTrainer::threadCount);
}

#include "pch.h"

#include "../JrBoostLib/StumpOptions.h"
#include "../JrBoostLib/StumpPredictor.h"
#include "../JrBoostLib/StumpTrainer.h"

#include "../JrBoostLib/BoostOptions.h"
#include "../JrBoostLib/BoostPredictor.h"
#include "../JrBoostLib/AdaBoostTrainer.h"
#include "../JrBoostLib/LogitBoostTrainer.h"

namespace py = pybind11;


PYBIND11_MODULE(jrboost, mod)
{
    mod.doc() = "The jrboost module implements the logit boost machine learning algorithm";

    py::register_exception<AssertionError>(mod, "AssertionError", PyExc_AssertionError);


    // Stump classes

    py::class_<StumpPredictor>{ mod, "StumpPredictor" }
        .def("variableCount", &StumpPredictor::variableCount)
        .def("predict", &StumpPredictor::predict, py::arg().noconvert());

    py::class_<StumpOptions>{ mod, "StumpOptions" }
        .def(py::init<>())
        .def_property("usedSampleRatio", &StumpOptions::usedSampleRatio, &StumpOptions::setUsedSampleRatio)
        .def_property("usedVariableRatio", &StumpOptions::usedVariableRatio, &StumpOptions::setUsedVariableRatio)
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

    py::class_<BoostOptions>{ mod, "BoostOptions" }
    .def(py::init<>())
        .def_property("iterationCount", &BoostOptions::iterationCount, &BoostOptions::setIterationCount)
        .def_property("eta", &BoostOptions::eta, &BoostOptions::setEta)
        .def_property("logStep", &BoostOptions::logStep, &BoostOptions::setLogStep)
        .def_property_readonly(
            "base",
            static_cast<StumpOptions& (BoostOptions::*)()>(&BoostOptions::base)
        );

    py::class_<AdaBoostTrainer>{ mod, "AdaBoostTrainer" }
        .def(py::init<RefXXf, ArrayXs>(), py::arg().noconvert(), py::arg())
        .def("train", &AdaBoostTrainer::train)
        .def("trainAndPredict", &AdaBoostTrainer::trainAndPredict);

    py::class_<LogitBoostTrainer>{ mod, "LogitBoostTrainer" }
        .def(py::init<CRefXXf, ArrayXs>(), py::arg().noconvert(), py::arg())
        .def("train", &LogitBoostTrainer::train)
        .def("trainAndPredict", &LogitBoostTrainer::trainAndPredict);
}

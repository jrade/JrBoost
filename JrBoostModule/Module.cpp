#include "pch.h"
#include "../JrBoostLib/TStatisticRank.h"
#include "../JrBoostLib/Loss.h"
#include "../JrBoostLib/BoostPredictor.h"
#include "../JrBoostLib/StumpOptions.h"
#include "../JrBoostLib/BoostOptions.h"
#include "../JrBoostLib/BoostTrainer.h"

namespace py = pybind11;


PYBIND11_MODULE(jrboost, mod)
{
    py::register_exception<AssertionError>(mod, "AssertionError", PyExc_AssertionError);

    mod.def("tStatisticRank", &tStatisticRank, py::arg().noconvert(), py::arg(), py::arg());
    mod.def("setNumThreads", &omp_set_num_threads);
    mod.def("setProfile", [](bool b) { PROFILE::doProfile = b; });


    // loss functions

    mod.def("linLoss", &linLoss).attr("name") = "lin-loss";
    mod.def("logLoss", &logLoss).attr("name") = "log-loss";
    mod.def("sqrtLoss", &sqrtLoss).attr("name") = "sqrt-loss";

    py::class_<AlphaLoss>{ mod, "AlphaLoss" }
        .def(py::init<double>())
        .def("__call__", &AlphaLoss::operator())
        .def_property_readonly("name", &AlphaLoss::name);

    py::class_<ErrorCount>{ mod, "ErrorCount" }
        .def(py::init<double>())
        .def("__call__", &ErrorCount::operator())
        .def_property_readonly("name", &ErrorCount::name);


    // PROFILE

    py::module profileMod = mod.def_submodule("PROFILE");

    py::enum_<PROFILE::CLOCK_ID>(profileMod, "CLOCK_ID")
        .value("MAIN", PROFILE::MAIN)
        .export_values();

    profileMod
        .def("PUSH", &PROFILE::PUSH)
        .def("POP", &PROFILE::POP, py::arg() = 0)
        .def("PRINT", &PROFILE::PRINT);


    // Stump classes

    py::class_<StumpOptions>{ mod, "StumpOptions" }
        .def(py::init<>())
        .def_property("usedSampleRatio", &StumpOptions::usedSampleRatio, &StumpOptions::setUsedSampleRatio)
        .def_property("usedVariableRatio", &StumpOptions::usedVariableRatio, &StumpOptions::setUsedVariableRatio)
        .def_property("topVariableCount", &StumpOptions::topVariableCount, &StumpOptions::setTopVariableCount)
        .def_property("minSampleWeight", &StumpOptions::minSampleWeight, &StumpOptions::setMinSampleWeight)
        .def_property("minNodeSize", &StumpOptions::minNodeSize, &StumpOptions::setMinNodeSize)
        .def_property("minNodeWeight", &StumpOptions::minNodeWeight, &StumpOptions::setMinNodeWeight)
        .def_property("isStratified", &StumpOptions::isStratified, &StumpOptions::setIsStratified);


    // Boost classes

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
        .def(py::init<ArrayXXf, ArrayXs>())
        .def("train", &BoostTrainer::train)
        .def("trainAndEval", &BoostTrainer::trainAndEval, py::call_guard<py::gil_scoped_release>());
        // BoostTrainer::trainAndEval() makes callbacks from OMP parallellized code.
        // These callbacks may be to Python functions that need to acquire the GIL.
        // If we don't relasee the GIL here it will be held by the master thread and the other threads will be blocked


    py::class_<BoostPredictor>{ mod, "BoostPredictor" }
        .def("variableCount", &BoostPredictor::variableCount)
        .def("predict", &BoostPredictor::predict);
}

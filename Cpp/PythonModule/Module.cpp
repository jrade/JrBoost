//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "../JrBoostLib/BoostOptions.h"
#include "../JrBoostLib/BoostPredictor.h"
#include "../JrBoostLib/BoostTrainer.h"
#include "../JrBoostLib/EnsemblePredictor.h"
#include "../JrBoostLib/Loss.h"
#include "../JrBoostLib/TTest.h"
#include "PyInterruptHandler.h"


PYBIND11_MODULE(_jrboostext, mod)
{
    namespace py = pybind11;

    currentInterruptHandler = &thePyInterruptHandler;

    py::register_exception<AssertionError>(mod, "AssertionError", PyExc_AssertionError);

    mod.def("setNumThreads", &omp_set_num_threads);


    // Predictors

    py::class_<Predictor, shared_ptr<Predictor>>{ mod, "Predictor" }
        .def("variableCount", &Predictor::variableCount)
        .def("predict", &Predictor::predict)
        .def("save", static_cast<void(Predictor::*)(const string&) const>(&Predictor::save))
        .def_static("load", static_cast<shared_ptr<Predictor>(*)(const string&)>(&Predictor::load));

    py::class_<EnsemblePredictor, shared_ptr<EnsemblePredictor>, Predictor>{ mod, "EnsemblePredictor" }
        .def(py::init<const vector<shared_ptr<Predictor>>&>());

    py::class_<BoostPredictor, shared_ptr<BoostPredictor>, Predictor>{ mod, "BoostPredictor" };


    // Boost trainer

    py::class_<BoostOptions>{ mod, "BoostOptions" }

        .def(py::init<>())

        .def_readonly_static("Ada", &BoostOptions::Ada)
        .def_readonly_static("Logit", &BoostOptions::Logit)

        .def_property("method", &BoostOptions::method, &BoostOptions::setMethod)
        .def_property("gamma", &BoostOptions::gamma, &BoostOptions::setGamma)
        .def_property("iterationCount", &BoostOptions::iterationCount, &BoostOptions::setIterationCount)
        .def_property("eta", &BoostOptions::eta, &BoostOptions::setEta)
        .def_property("minAbsSampleWeight", &BoostOptions::minAbsSampleWeight, &BoostOptions::setMinAbsSampleWeight)
        .def_property("minRelSampleWeight", &BoostOptions::minRelSampleWeight, &BoostOptions::setMinRelSampleWeight)
        .def_property("fastExp", &BoostOptions::fastExp, &BoostOptions::setFastExp)

        .def_property("usedSampleRatio", &BoostOptions::usedSampleRatio, &BoostOptions::setUsedSampleRatio)
        .def_property("usedVariableRatio", &BoostOptions::usedVariableRatio, &BoostOptions::setUsedVariableRatio)
        .def_property("topVariableCount", &BoostOptions::topVariableCount, &BoostOptions::setTopVariableCount)
        .def_property("minNodeSize", &BoostOptions::minNodeSize, &BoostOptions::setMinNodeSize)
        .def_property("minNodeWeight", &BoostOptions::minNodeWeight, &BoostOptions::setMinNodeWeight)
        .def_property("isStratified", &BoostOptions::isStratified, &BoostOptions::setIsStratified)

        .def("__copy__", [](const BoostOptions& bOpt) { return BoostOptions(bOpt); });


    py::class_<BoostTrainer>{ mod, "BoostTrainer" }
        .def(py::init<ArrayXXf, ArrayXs, optional<ArrayXd>>(), py::arg(), py::arg(), py::arg("weights") = optional<ArrayXd>())
        .def("train", &BoostTrainer::train)
        .def("trainAndEval", &BoostTrainer::trainAndEval, py::call_guard<py::gil_scoped_release>());
        // BoostTrainer::trainAndEval() makes callbacks from OMP parallellized code.
        // These callbacks may be to Python functions that need to acquire the GIL.
        // If we don't release the GIL here it will be held by the master thread and the other threads will be blocked.


    // Loss functions

    mod.def("errorCount", &errorCount, py::arg(), py::arg(), py::arg("threshold") = 0.5);
    mod.def("linLoss", &linLoss);
    mod.def("logLoss", &logLoss, py::arg(), py::arg(), py::arg("gamma") = 0.1);
    mod.def("auc", &auc);
    mod.def("negAuc", &negAuc);

    py::class_<ErrorCount>{ mod, "ErrorCount" }
    .def(py::init<double>())
        .def("__call__", &ErrorCount::operator())
        .def_property_readonly("name", &ErrorCount::name);

    py::class_<LogLoss>{ mod, "LogLoss" }
    .def(py::init<double>())
        .def("__call__", &LogLoss::operator())
        .def_property_readonly("name", &LogLoss::name);


    // Statistical tests

    py::enum_<TestDirection>(mod, "TestDirection")
        .value("Up", TestDirection::Up)
        .value("Down", TestDirection::Down)
        .value("Any", TestDirection::Any);

    mod.def("tTestRank", &tTestRank,
        py::arg(), py::arg(), py::arg("samples") = optional<CRefXs>(), py::arg("direction") = TestDirection::Any);


    // Profiling

    mod.def("setProfile", [](bool b) { PROFILE::doProfile = b; });

    py::module profileMod = mod.def_submodule("PROFILE");

    py::enum_<PROFILE::CLOCK_ID>(profileMod, "CLOCK_ID")
        .value("MAIN", PROFILE::MAIN)
        .export_values();

    profileMod
        .def("PUSH", &PROFILE::PUSH)
        .def("POP", &PROFILE::POP, py::arg() = 0)
        .def("PRINT", &PROFILE::PRINT);
}

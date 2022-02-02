//  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "../JrBoostLib/BoostTrainer.h"
#include "../JrBoostLib/FTest.h"
#include "../JrBoostLib/Loss.h"
#include "../JrBoostLib/Paralleltrain.h"
#include "../JrBoostLib/Predictor.h"
#include "../JrBoostLib/TTest.h"
#include "../JrBoostLib/TopScoringPairs.h"
#include "../JrBoostLib/TreeTrainerBuffers.h"
#include "PyBoostOptions.h"
#include "PyInterruptHandler.h"


PYBIND11_MODULE(_jrboost, mod)
{
    namespace py = pybind11;

    omp_set_nested(true);

    ::currentInterruptHandler = &thePyInterruptHandler;

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const AssertionError& e) {
            PyErr_SetString(PyExc_AssertionError, e.what());
        }
    });


    // Predictor

    py::class_<Predictor, shared_ptr<Predictor>>{mod, "Predictor"}
        .def("predict", [](shared_ptr<Predictor> predictor, CRefXXfc inData) { return predictor->predict(inData); })
        .def("predictOne", &Predictor::predictOne)
        .def("variableCount", &Predictor::variableCount)
        .def("variableWeights", &Predictor::variableWeights)
        .def("reindexVariables", &Predictor::reindexVariables)
        .def("save", py::overload_cast<const string&>(&Predictor::save, py::const_))
        .def_static("load", py::overload_cast<const string&>(&Predictor::load))
        .def_static("createEnsemble", &EnsemblePredictor::createInstance)
        .def_static("createUnion", &UnionPredictor::createInstance)
        //.def_static(
        //    "createShifted", py::overload_cast<shared_ptr<Predictor>, double,
        //    double>(&ShiftPredictor::createInstance))
        .def("__repr__", [](const Predictor&) { return "<jrboost.Predictor>"; })
        .def(py::pickle(
            [](const Predictor& pred) {
                stringstream ss;
                ss.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
                pred.save(ss);
                return static_cast<py::bytes>(ss.str());
            },
            [](const py::bytes& b) {
                stringstream ss(static_cast<string>(b));
                ss.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
                return Predictor::load(ss);
            }));


    // Boost trainer

    py::class_<BoostTrainer>{mod, "BoostTrainer"}
        .def(
            py::init<ArrayXXfc, ArrayXu8, optional<ArrayXd>, optional<ArrayXu8>>(), py::arg(), py::arg(), py::kw_only(),
            py::arg("weights") = std::nullopt, py::arg("strata") = std::nullopt)
        .def("train", [](const BoostTrainer& trainer, const BoostOptions& opt) { return trainer.train(opt); })
        .def("__repr__", [](const BoostTrainer&) { return "<jrboost.BoostTrainer>"; });

    mod.def("getDefaultBoostParam", []() { return BoostOptions(); });

    mod.def("parallelTrain", &parallelTrain);
    mod.def("parallelTrainAndPredict", &parallelTrainAndPredict);
    mod.def(
        "parallelTrainAndEval", &parallelTrainAndEval, py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg("weights") = std::nullopt, py::call_guard<py::gil_scoped_release>());

    // parallelTrainAndEval() makes callbacks from multi-threaded code.
    // These callbacks may be to Python functions that need to acquire the GIL.
    // If we don't release the GIL here it will be held by the master thread and the other threads will be blocked.


    // Loss functions

    mod.def("linLoss", &linLoss, py::arg(), py::arg(), py::arg("weights") = std::nullopt);
    mod.def("auc", &auc, py::arg(), py::arg(), py::arg("weights") = std::nullopt);
    mod.def("aoc", &aoc, py::arg(), py::arg(), py::arg("weights") = std::nullopt);
    mod.def("negAuc", &negAuc, py::arg(), py::arg(), py::arg("weights") = std::nullopt);

    py::class_<LogLoss>{mod, "LogLoss"}
        .def(py::init<double>())
        .def("__call__", &LogLoss::operator(), py::arg(), py::arg(), py::arg("weights") = std::nullopt)
        .def_property_readonly("name", &LogLoss::name);


    // Statistical tests

    py::enum_<TestDirection>(mod, "TestDirection")
        .value("Up", TestDirection::Up)
        .value("Down", TestDirection::Down)
        .value("Any", TestDirection::Any);

    mod.def("tStatistic", &tStatistic, py::arg().noconvert(), py::arg(), py::arg("samples") = std::nullopt);

    mod.def(
        "tTestRank", &tTestRank, py::arg().noconvert(), py::arg(), py::arg("samples") = std::nullopt,
        py::arg("direction") = TestDirection::Any);

    mod.def("fStatistic", &fStatistic, py::arg().noconvert(), py::arg(), py::arg("samples") = std::nullopt);

    mod.def("fTestRank", &fTestRank, py::arg().noconvert(), py::arg(), py::arg("samples") = std::nullopt);

    mod.def(
        "topScoringPairs", &topScoringPairs, py::arg().noconvert(), py::arg(), py::arg(),
        py::arg("samples") = std::nullopt);

    mod.def("filterPairs", &filterPairs);
    

    // Other

    mod.def("getThreadCount", &omp_get_max_threads);
    mod.def("setThreadCount", &omp_set_num_threads);

    mod.def("bufferSize", &TreeTrainerBuffers::bufferSize);
    mod.def("clearBuffers", &TreeTrainerBuffers::freeBuffers);

    mod.attr("eigenVersion") = py::str(theEigenVersion);
    mod.attr("pybind11Version") = py::str(thePybind11Version);


    // Profiling

    py::module profileMod = mod.def_submodule("PROFILE");

    profileMod.def("START", &PROFILE::START)
        .def("STOP", &PROFILE::STOP)
        .def("PUSH", &PROFILE::PUSH)
        .def("POP", &PROFILE::POP, py::arg() = 0);

    py::enum_<PROFILE::CLOCK_ID>(profileMod, "CLOCK_ID")
        .value("MAIN", PROFILE::MAIN)
        .value("TEST1", PROFILE::TEST1)
        .value("TEST2", PROFILE::TEST2)
        .value("TEST3", PROFILE::TEST3)
        .value("TEST4", PROFILE::TEST4)
        .value("TEST5", PROFILE::TEST5)
        .export_values();
}

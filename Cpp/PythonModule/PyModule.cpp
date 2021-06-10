//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"
#include "../JrBoostLib/BoostOptions.h"
#include "../JrBoostLib/BoostPredictor.h"
#include "../JrBoostLib/BoostTrainer.h"
#include "../JrBoostLib/EnsemblePredictor.h"
#include "../JrBoostLib/Paralleltrain.h"
#include "../JrBoostLib/Loss.h"
#include "../JrBoostLib/TTest.h"
#include "../JrBoostLib/FTest.h"
#include "PyBoostOptions.h"
#include "PyInterruptHandler.h"


inline py::bytes predictorGetState(const Predictor& pred);
template<typename T> inline shared_ptr<T> predictorSetState(const py::bytes& s);


PYBIND11_MODULE(_jrboostext, mod)
{
    namespace py = pybind11;

    currentInterruptHandler = &thePyInterruptHandler;

    py::register_exception_translator(
        [] (std::exception_ptr p) {
            try {
                if (p) std::rethrow_exception(p);
            }
            catch (const AssertionError& e) {
                PyErr_SetString(PyExc_AssertionError, e.what());
            }
        }
    );

    mod.def("getThreadCount", &omp_get_max_threads);
    mod.def("setThreadCount", &omp_set_num_threads);
    mod.attr("eigenVersion") = py::str(eigenVersion);
    mod.attr("pybind11Version") = py::str(pybind11Version);


    // Predictors

    py::class_<Predictor, shared_ptr<Predictor>>{ mod, "Predictor" }
        .def("variableCount", &Predictor::variableCount)
        .def("predict", &Predictor::predict)
        .def("save", py::overload_cast<const string&>(&Predictor::save, py::const_))
        .def_static("load", py::overload_cast<const string&>(&Predictor::load));

    py::class_<EnsemblePredictor, shared_ptr<EnsemblePredictor>, Predictor>{ mod, "EnsemblePredictor" }
        .def(py::init<const vector<shared_ptr<Predictor>>&>())
        .def(py::pickle(&predictorGetState, &predictorSetState<EnsemblePredictor>));

    py::class_<BoostPredictor, shared_ptr<BoostPredictor>, Predictor>{ mod, "BoostPredictor" }
        .def(py::pickle(&predictorGetState, &predictorSetState<BoostPredictor>));


    // Boost trainer

    py::class_<BoostTrainer>{ mod, "BoostTrainer" }
        .def(
            py::init<ArrayXXf, ArrayXs, optional<ArrayXd>>(),
            py::arg(), py::arg(), py::arg("weights") = optional<ArrayXd>()
        )
        .def("train", &BoostTrainer::train);

    mod.def("getDefaultBoostParam", []() { return BoostOptions(); });

    mod.def("parallelTrain", &parallelTrain);
    mod.def("parallelTrainAndPredict", &parallelTrainAndPredict);
    mod.def("parallelTrainAndEval", &parallelTrainAndEval, py::call_guard<py::gil_scoped_release>());

    // parallelTrainAndEval() makes callbacks from multi-threaded code.
    // These callbacks may be to Python functions that need to acquire the GIL.
    // If we don't relasee the GIL here it will be held by the master thread and the other threads will be blocked


    // Loss functions

    mod.def("errorCount", &errorCount, py::arg(), py::arg(), py::arg("threshold") = 0.5);
    mod.def("accuracy", &accuracy, py::arg(), py::arg(), py::arg("threshold") = 0.5);
    mod.def("linLoss", &linLoss);
    mod.def("logLoss", &logLoss, py::arg(), py::arg(), py::arg("gamma") = 0.1);
    mod.def("auc", &auc);
    mod.def("aoc", &aoc);
    mod.def("negAuc", &negAuc);

    py::class_<ErrorCount>{ mod, "ErrorCount" }
        .def(py::init<double>())
        .def("__call__", &ErrorCount::operator())
        .def_property_readonly("name", &ErrorCount::name);

    py::class_<Accuracy>{ mod, "Accuracy" }
        .def(py::init<double>())
        .def("__call__", &Accuracy::operator())
        .def_property_readonly("name", &Accuracy::name);

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

    mod.def("fTestRank", &fTestRank,
        py::arg(), py::arg(), py::arg("samples") = optional<CRefXs>());


    // Profiling

    py::module profileMod = mod.def_submodule("PROFILE");

    profileMod
        // high level API
        .def("START", &PROFILE::START)
        .def("STOP", &PROFILE::STOP)
        // low level API
        .def("GET_ENABLED", []() { return PROFILE::ENABLED; })
        .def("SET_ENABLED", [](bool b) { PROFILE::ENABLED = b; })
        .def("PUSH", &PROFILE::PUSH)
        .def("POP", &PROFILE::POP, py::arg() = 0)
        .def("RESULT", &PROFILE::RESULT);

    py::enum_<PROFILE::CLOCK_ID>(profileMod, "CLOCK_ID")
        .value("MAIN", PROFILE::MAIN)
        .value("TEST1", PROFILE::TEST1)
        .value("TEST2", PROFILE::TEST2)
        .value("TEST3", PROFILE::TEST3)
        .export_values();
}

//----------------------------------------------------------------------------------------------------------------------

inline py::bytes predictorGetState(const Predictor& pred)
{
    stringstream ss;
    ss.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    pred.save(ss);
    return static_cast<py::bytes>(ss.str());
}

template<typename T>
inline shared_ptr<T> predictorSetState(const py::bytes& b)
{
    stringstream ss(static_cast<string>(b));
    ss.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    shared_ptr<Predictor> pred = Predictor::load(ss);
    shared_ptr<T> tPred = std::dynamic_pointer_cast<T>(pred);
    ASSERT(tPred);
    return tPred;
}

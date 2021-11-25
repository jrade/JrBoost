//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#include "pch.h"

#include "PyBoostOptions.h"
#include "PyInterruptHandler.h"
#include "../JrBoostLib/BoostTrainer.h"
#include "../JrBoostLib/FTest.h"
#include "../JrBoostLib/Loss.h"
#include "../JrBoostLib/Paralleltrain.h"
#include "../JrBoostLib/Predictor.h"
#include "../JrBoostLib/TreeTrainerBase.h"
#include "../JrBoostLib/TTest.h"


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


    // Predictor

    py::class_<Predictor, shared_ptr<Predictor>>{ mod, "Predictor" }
        .def("predict", &Predictor::predict)
        .def("variableCount", &Predictor::variableCount)
        .def("variableWeights", &Predictor::variableWeights)
        .def("reindexVariables", &Predictor::reindexVariables)
        .def("save", py::overload_cast<const string&>(&Predictor::save, py::const_))
        .def_static("load", py::overload_cast<const string&>(&Predictor::load))
        .def_static("createEnsemble",
            [] (const vector<shared_ptr<const Predictor>> predictors) {
                return std::const_pointer_cast<Predictor>(EnsemblePredictor::createInstance(predictors));
            }
        )
        .def_static("createUnion",
            [] (const vector<shared_ptr<const Predictor>> predictors) {
                return std::const_pointer_cast<Predictor>(UnionPredictor::createInstance(predictors));
            }
        )
        .def("__repr__",
            [] (const Predictor&) {
                return "<jrboost.Predictor>";
            }
        )
        .def(py::pickle(
            [] (const Predictor& pred) {
                stringstream ss;
                ss.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
                pred.save(ss);
                return static_cast<py::bytes>(ss.str());
            },
            [] (const py::bytes& b) {
                stringstream ss(static_cast<string>(b));
                ss.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
                return std::const_pointer_cast<Predictor>(Predictor::load(ss));
            }
        ));


    // Boost trainer

    py::class_<BoostTrainer>{ mod, "BoostTrainer" }
        .def(
            py::init<ArrayXXfc, ArrayXu8, optional<ArrayXd>>(),
            py::arg(), py::arg(), py::arg("weights") = optional<ArrayXd>()
        )
        .def("train",
            [] (const BoostTrainer& trainer, const BoostOptions& opt) {
            return std::const_pointer_cast<Predictor>(trainer.train(opt));
        })
        .def("__repr__", [] (const BoostTrainer&) { return "<jrboost.BoostTrainer>"; });

    mod.def("getDefaultBoostParam", [] () { return BoostOptions(); });

    mod.def("parallelTrain", &parallelTrain,
        py::arg(), py::arg());
    mod.def("parallelTrainAndPredict", &parallelTrainAndPredict,
        py::arg(), py::arg(), py::arg());
    mod.def("parallelTrainAndEval", &parallelTrainAndEval,
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::call_guard<py::gil_scoped_release>());
    mod.def("parallelTrainAndEvalWeighted", &parallelTrainAndEvalWeighted,
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::call_guard<py::gil_scoped_release>());

    // parallelTrainAndEval() makes callbacks from multi-threaded code.
    // These callbacks may be to Python functions that need to acquire the GIL.
    // If we don't relasee the GIL here it will be held by the master thread and the other threads will be blocked


    // Loss functions

    mod.def("linLoss", &linLoss);
    mod.def("linLossWeighted", &linLossWeighted);
    mod.def("logLoss", &logLoss, py::arg(), py::arg(), py::arg("gamma") = 0.001);
    mod.def("logLossWeighted", &logLossWeighted, py::arg(), py::arg(), py::arg(), py::arg("gamma") = 0.001);
    mod.def("auc", &auc);
    mod.def("aucWeighted", &aucWeighted);
    mod.def("aoc", &aoc);
    mod.def("aocWeighted", &aocWeighted);
    mod.def("negAuc", &negAuc);
    mod.def("negAucWeighted", &negAucWeighted);

    py::class_<LogLoss>{ mod, "LogLoss" }
        .def(py::init<double>())
        .def("__call__", &LogLoss::operator())
        .def_property_readonly("name", &LogLoss::name);

    py::class_<LogLossWeighted>{ mod, "LogLossWeighted" }
        .def(py::init<double>())
        .def("__call__", &LogLossWeighted::operator())
        .def_property_readonly("name", &LogLossWeighted::name);


    // Statistical tests

    py::enum_<TestDirection>(mod, "TestDirection")
        .value("Up", TestDirection::Up)
        .value("Down", TestDirection::Down)
        .value("Any", TestDirection::Any);

    mod.def("tStatistic", &tStatistic,
        py::arg().noconvert(), py::arg(),
        py::arg("samples") = optional<vector<size_t>>());

    mod.def("tTestRank", &tTestRank,
        py::arg().noconvert(), py::arg(),
        py::arg("samples") = optional<vector<size_t>>(),
        py::arg("direction") = TestDirection::Any);

    mod.def("fStatistic", &fStatistic,
        py::arg().noconvert(), py::arg(),
        py::arg("samples") = optional<vector<size_t>>());

    mod.def("fTestRank", &fTestRank,
        py::arg().noconvert(), py::arg(),
        py::arg("samples") = optional<vector<size_t>>());


    // Other

    mod.def("getThreadCount", &omp_get_max_threads);
    mod.def("setThreadCount", &omp_set_num_threads);

    mod.def("getParallelTree", [] () { return ::globParallelTree; });
    mod.def("setParallelTree", [] (bool b) { ::globParallelTree = b; });

    mod.def("getOuterThreadCount", [] () { return ::globOuterThreadCount; });
    mod.def("setOuterThreadCount", [] (size_t n) { ::globOuterThreadCount = n; });

    mod.def("bufferSize", &TreeTrainerBase::bufferSize);
    mod.def("clearBuffers", &TreeTrainerBase::freeBuffers);

    mod.attr("eigenVersion") = py::str(theEigenVersion);
    mod.attr("pybind11Version") = py::str(thePybind11Version);


    // Profiling

    py::module profileMod = mod.def_submodule("PROFILE");

    profileMod
        .def("START", &PROFILE::START)
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

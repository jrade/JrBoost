#include "pch.h"
#include "../Tools/Util.h"
#include "../JrBoostLib/AbstractPredictor.h"
#include "../JrBoostLib/StumpOptions.h"
#include "../JrBoostLib/StumpTrainer.h"
#include "../JrBoostLib/BoostOptions.h"
#include "../JrBoostLib/BoostTrainer.h"

namespace py = pybind11;


PYBIND11_MODULE(jrboost, mod)
{
    py::register_exception<AssertionError>(mod, "AssertionError", PyExc_AssertionError);

    mod.def("linLoss", &linLoss);

    mod.def(
        "tStatisticRank",
        &tStatisticRank,
        py::arg().noconvert(),
        py::arg().noconvert(),
        py::arg().noconvert()
    );
    mod.def("ompSetNumThreads", &omp_set_num_threads);


    // PROFILE submodule

    py::module clockMod = mod.def_submodule("PROFILE");

    py::enum_<PROFILE::CLOCK_ID>(clockMod, "CLOCK_ID")
        .value("MAIN", PROFILE::MAIN)
        .export_values();

    clockMod
        .def("PUSH", &PROFILE::PUSH)
        .def("POP", &PROFILE::POP, py::arg() = 0)
        .def("PRINT", &PROFILE::PRINT);


    // Abstract predictor

    py::class_<AbstractPredictor>{ mod, "Predictor" }
        .def("variableCount", &AbstractPredictor::variableCount)
        .def(
            "predict", 
            static_cast<ArrayXd (AbstractPredictor::*)(CRefXXf inData) const>(&AbstractPredictor::predict),
            py::arg().noconvert()
        );


    // Stump classes

    py::class_<StumpOptions>{ mod, "StumpOptions" }
    .def(py::init<>())
        .def_property("usedSampleRatio", &StumpOptions::usedSampleRatio, &StumpOptions::setUsedSampleRatio)
        .def_property("usedVariableRatio", &StumpOptions::usedVariableRatio, &StumpOptions::setUsedVariableRatio)
        .def_property("topVariableCount", &StumpOptions::topVariableCount, &StumpOptions::setTopVariableCount)
        .def_property("minNodeSize", &StumpOptions::minNodeSize, &StumpOptions::setMinNodeSize)
        .def_property("minNodeWeight", &StumpOptions::minNodeWeight, &StumpOptions::setMinNodeWeight)
        .def_property("isStratified", &StumpOptions::isStratified, &StumpOptions::setIsStratified);

    py::class_<StumpTrainer>{ mod, "StumpTrainer" }
        .def(py::init<RefXXf, RefXs>(), py::arg().noconvert(), py::arg().noconvert())
        .def("train", &StumpTrainer::train);


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
        .def(py::init<RefXXf, RefXs>(), py::arg().noconvert(), py::arg())
        .def("train", &BoostTrainer::train)
        .def("trainAndEval", &BoostTrainer::trainAndEval);
}

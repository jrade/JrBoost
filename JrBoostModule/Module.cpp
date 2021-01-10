#include "pch.h"

#include "../JrBoostLib/StubPredictor.h"
#include "../JrBoostLib/StubTrainer.h"
#include "../JrBoostLib/StubOptions.h"

#include "../JrBoostLib/BoostPredictor.h"
#include "../JrBoostLib/AdaBoostTrainer.h"
#include "../JrBoostLib/AdaBoostOptions.h"

#pragma warning( disable : 26444 )

namespace py = pybind11;

PYBIND11_MODULE(jrboost, mod)
{
    mod.doc() = "The jrboost module implements the logit boost machine learning algorithm";

    py::register_exception<AssertionError>(mod, "AssertionError", PyExc_AssertionError);

    // Abstract classes

    py::class_<AbstractPredictor> { mod, "AbstractPredictor" }
        .def("variableCount", &AbstractPredictor::variableCount)
        .def("predict", &AbstractPredictor::predict);
    
    py::class_<AbstractTrainer>{ mod, "AbstractTrainer" }
        .def("setInData", &AbstractTrainer::setInData, py::arg{}.noconvert())
        .def("setOutData", &AbstractTrainer::setOutData)
        .def("setWeights", &AbstractTrainer::setWeights)
        .def("setOptions", &AbstractTrainer::setOptions)
        .def("train", &AbstractTrainer::train);
    
    py::class_<AbstractOptions>{ mod, "AbstractOptions" }
        .def("createTrainer", &AbstractOptions::createTrainer);
 
    // Stub classes
    
    py::class_<StubPredictor, AbstractPredictor>{ mod, "StubPredictor" };

    py::class_<StubTrainer, AbstractTrainer>{ mod, "StubTrainer" }
        .def(py::init<>());
    
    py::class_<StubOptions, AbstractOptions>{ mod, "StubOptions" }
        .def(py::init<>())
        .def_property("usedSampleRatio", &StubOptions::usedSampleRatio, &StubOptions::setUsedSampleRatio)
        .def_property("usedVariableRatio", &StubOptions::usedVariableRatio, &StubOptions::setUsedVariableRatio)
        .def_property("highPrecision", &StubOptions::highPrecision, &StubOptions::setHighPrecision)
        .def_property("profile", &StubOptions::profile, &StubOptions::setProfile);

    // Boost classes

        py::class_<BoostPredictor, AbstractPredictor>{ mod, "BoostPredictor" };

        py::class_<AdaBoostTrainer, AbstractTrainer>{ mod, "AdaBoostTrainer" }
        .def(py::init<>());

        py::class_<AdaBoostOptions, AbstractOptions>{ mod, "AdaBoostOptions" }
        .def(py::init<>())
            .def_property("iterationCount", &AdaBoostOptions::iterationCount, &AdaBoostOptions::setIterationCount)
            .def_property("eta", &AdaBoostOptions::eta, &AdaBoostOptions::setEta)
            .def_property("highPrecision", &AdaBoostOptions::highPrecision, &AdaBoostOptions::setHighPrecision)
            //.def_property("clamp", &AdaBoostOptions::clamp, &AdaBoostOptions::setClamp)
            .def_property("baseOptions", &AdaBoostOptions::baseOptions, &AdaBoostOptions::setBaseOptions);
}

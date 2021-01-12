#include "pch.h"

#include "../JrBoostLib/StumpPredictor.h"
#include "../JrBoostLib/StumpTrainer.h"
#include "../JrBoostLib/StumpOptions.h"

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
 
    // Stump classes
    
    py::class_<StumpPredictor, AbstractPredictor>{ mod, "StumpPredictor" };

    py::class_<StumpTrainer, AbstractTrainer>{ mod, "StumpTrainer" }
        .def(py::init<>());
    
    py::class_<StumpOptions, AbstractOptions>{ mod, "StumpOptions" }
        .def(py::init<>())
        .def_property("usedSampleRatio", &StumpOptions::usedSampleRatio, &StumpOptions::setUsedSampleRatio)
        .def_property("usedVariableRatio", &StumpOptions::usedVariableRatio, &StumpOptions::setUsedVariableRatio)
        .def_property("highPrecision", &StumpOptions::highPrecision, &StumpOptions::setHighPrecision)
        .def_property("profile", &StumpOptions::profile, &StumpOptions::setProfile);

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

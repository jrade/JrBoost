#include "pch.h"
#include "../JrBoostLib/StubPredictor.h"
#include "../JrBoostLib/StubTrainer.h"
#include "../JrBoostLib/StubOptions.h"

namespace py = pybind11;

PYBIND11_MODULE(jrboost, mod)
{
    mod.doc() = "The jrboost module implements the logit boost machine learning algorithm";

    // Abstract classes

    py::class_<AbstractPredictor>(mod, "AbstractPredictor")
        .def("variableCount", &AbstractPredictor::variableCount)
        .def("predict", &AbstractPredictor::predict);
    
    py::class_<AbstractTrainer>(mod, "AbstractTrainer")
        .def("setInData", &AbstractTrainer::setInData)
        .def("setOutData", &AbstractTrainer::setOutData)
        .def("setWeights", &AbstractTrainer::setWeights)
        .def("setOptions", &AbstractTrainer::setOptions)
        .def("train", &AbstractTrainer::train);
    
    py::class_<AbstractOptions>(mod, "AbstractOptions")
        .def("createTrainer", &AbstractOptions::createTrainer);
 
    // Stub classes
    
    py::class_<StubPredictor, AbstractPredictor>(mod, "StubPredictor");

    py::class_<StubTrainer, AbstractTrainer>(mod, "StubTrainer")
        .def(py::init<>());
    
    py::class_<StubOptions, AbstractOptions>(mod, "StubOptions")
        .def(py::init<>())
        .def_readwrite("usedSampleRatio", &StubOptions::usedSampleRatio)
        .def_readwrite("usedVariableRatio", &StubOptions::usedVariableRatio);
}

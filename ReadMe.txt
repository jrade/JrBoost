To build JrBoost you need:

	a 64-bit C++17 compiler
	a Python 3 installation with NumPy
	the Eigen and pybind11 C++ libraries

JrBoost has been built with:

  Microsoft Visual Studio 2019
  Anaconda version 4.9.2 (includes Python version 3.8.5 and NumPy version 1.19.2)
  Eigen version 3.4.0
  pybind11 version 2.7.1

The code comes with Visual Studio 2019 solution and project files.
These use the environment variables EIGEN_DIR, PYBIND11_DIR and PYTHON_DIR
to find the C++ library and the python installation root directories.
